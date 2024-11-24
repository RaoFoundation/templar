# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Global imports.
import argparse
import asyncio
import bittensor as bt
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
import random
import sys
from transformers import LlamaForCausalLM
import templar as tplr

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class DecayMiner:

    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Decay Miner script')
        parser.add_argument('--project', type=str, default='templar', help='Optional wandb project name')
        parser.add_argument('--netuid', type=int, default=3, help='Bittensor network UID.')
        parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
        parser.add_argument('--actual_batch_size', type=int, default=8, help='Training batch size per accumulation.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
        parser.add_argument('--remote', action='store_true', help='Connect to other buckets')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        parser.add_argument('--sync_state', action='store_true', help='Syncs the model state by pulling from the history.')
        parser.add_argument('--test', action='store_true', help='Run on test network')
        parser.add_argument('--local', action='store_true', help='Run on local network')
        parser.add_argument('--autoupdate', action='store_true', help='Enable automatic updates')
        parser.add_argument('--process_name', type=str, help='The name of the PM2 process')
        parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to save/load the checkpoint.')

        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)

        config = bt.config(parser)
        if config.test:
            config.subtensor.network = 'test'
            config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'
        elif config.local:
            config.subtensor.network = 'local'
            config.subtensor.chain_endpoint = 'ws://127.0.0.1:9944'
        if config.debug: tplr.debug()
        if config.trace: tplr.trace()
        tplr.validate_bucket_or_exit(config.bucket)
        if config.autoupdate:
            autoupdater = tplr.AutoUpdate(process_name=config.process_name, bucket_name=config.bucket)
            autoupdater.start()
        return config

    def __init__(self):
        # Init config.
        self.config = DecayMiner.config()
        tplr.logger.info('\n' + '-' * 40 + ' Config ' + '-' * 40)
        tplr.logger.info(self.config)

        # Init bittensor objects.
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            tplr.logger.error(f'\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}[/bold]. You need to register first.\n')
            sys.exit()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        tplr.logger.info('\n' + '-' * 40 + ' Objects ' + '-' * 40)
        tplr.logger.info(f'\nWallet: {self.wallet}\nSubtensor: {self.subtensor}\nMetagraph: {self.metagraph}\nUID: {self.uid}')

        # Init hyperparameters.
        self.hparams = tplr.load_hparams()

        # Set device
        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')

        # Init model.
        tplr.logger.info('\n' + '-' * 40 + ' Model ' + '-' * 40)
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        self.model = LlamaForCausalLM(config=self.hparams.model_config)
        self.model.to(self.device)
        self.model.train()

        # Init optimizer.
        self.optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate)

        # Initialize scheduler for decay phase only.
        self.scheduler = tplr.get_wsd_scheduler(
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_stable_steps=0,
            num_decay_steps=self.hparams.num_decay_steps,
            last_epoch=-1  # Start fresh
        )

        # Initialize global step.
        self.global_step = 0

    async def run(self):
        # Get the latest available model block.
        latest_block = await self.get_latest_model_block()
        if latest_block is None:
            tplr.logger.error("No model available for decay phase.")
            return

        # Download the latest model state dict.
        model_state = await tplr.download_master(bucket=self.config.bucket, wallet=self.wallet, current_block=latest_block)
        if model_state is None:
            tplr.logger.error("Failed to download model for decay phase.")
            return

        # Load the state dict into the model.
        self.model.load_state_dict(model_state)
        self.model.train()

        # Prepare the dataset.
        tplr.logger.info("Preparing dataset for decay phase.")
        train_dataset = await self.prepare_dataset()

        # Run the decay phase.
        tplr.logger.info("Starting decay phase training.")
        for epoch in range(self.hparams.num_decay_epochs):
            for batch in train_dataset:
                input_ids = torch.tensor(batch, dtype=torch.long).to(self.device)
                labels = input_ids.clone()
                labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)

                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

 
                tplr.logger.info(f"Global Step: {self.global_step}, Loss: {loss.item():.4f}, Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")

        tplr.logger.info("Decay phase completed.")

        # Optionally, save or upload the final model.
        await self.save_final_model()

    async def get_latest_model_block(self):
        """
        Retrieves the block number of the latest uploaded master model.
        """
        # Logic to find the latest block number with an uploaded model
        # Assuming models are stored with filenames like 'master-<hotkey>-block<block>-v<version>.pt.gz'
        session = tplr.get_session()
        async with session.create_client(
            's3',
            region_name='us-east-1',
            config=tplr.client_config,
            aws_access_key_id=tplr.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=tplr.AWS_SECRET_ACCESS_KEY
        ) as s3_client:
            try:
                prefix = f'master-{self.wallet.hotkey.ss58_address}-block'
                response = await s3_client.list_objects_v2(Bucket=self.config.bucket, Prefix=prefix)
                if 'Contents' not in response:
                    tplr.logger.error("No models found in the bucket.")
                    return None

                # Extract block numbers from filenames
                blocks = []
                for obj in response['Contents']:
                    key = obj['Key']
                    parts = key.split('-block')
                    if len(parts) > 1:
                        block_part = parts[1].split('-v')[0]
                        try:
                            block_num = int(block_part)
                            blocks.append(block_num)
                        except ValueError:
                            continue

                if not blocks:
                    tplr.logger.error("No valid model files found.")
                    return None

                latest_block = max(blocks)
                tplr.logger.info(f"Latest model block found: {latest_block}")
                return latest_block

            except Exception as e:
                tplr.logger.exception(f"Failed to list models in bucket: {e}")
                return None

    async def prepare_dataset(self):
        """
        Prepares the dataset for training.

        Returns:
            train_dataset: Iterable dataset for training.
        """

        pages = await tplr.dataset.DatasetLoader.next_pages(
            offset=0,  # Starting offset
            n_pages=self.hparams.num_decay_steps * self.hparams.validator_window_eval_size,
            seed=42  # Fixed seed for reproducibility
        )
        train_dataset = await tplr.dataset.DatasetLoader.create(
            batch_size=self.config.actual_batch_size,
            sequence_length=self.hparams.sequence_length,
            pages_info=pages,
            tokenizer=self.hparams.tokenizer
        )
        return train_dataset

    async def save_final_model(self):
        """
        Saves the final model after decay phase.
        """
        final_model_path = f'final_model-{self.wallet.hotkey.ss58_address}.pt'
        torch.save(self.model.state_dict(), final_model_path)
        tplr.logger.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    decay_miner = DecayMiner()
    asyncio.run(decay_miner.run())