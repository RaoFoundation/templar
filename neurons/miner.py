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
# fmt: off

# Global imports.
import sys
import time 
import wandb
import torch
import random
import asyncio
import argparse
import threading
import numpy as np
from tqdm import tqdm
import bittensor as bt
import torch.optim as optim
from transformers import LlamaForCausalLM
from rich.markup import escape
import os

# Import local files.
import templar as tplr

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Miner:

    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Miner script')
        parser.add_argument('--project', type=str, default='templar', help='Optional wandb project name')
        parser.add_argument('--netuid', type=int, default=3, help='Bittensor network UID.')
        parser.add_argument('--actual_batch_size', type=int, default=8, help='Training batch size per accumulation.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        parser.add_argument('--random', action='store_true', help='Train on random')
        parser.add_argument('--sync_state', action='store_true', help='Syncs the model state by pulling from the history.')
        parser.add_argument('--baseline', action='store_true', help='Dont perform syncing with other peers, just train.')
        parser.add_argument('--test', action='store_true', help='Run on test network')
        parser.add_argument('--local', action='store_true', help='Run on local network')
        parser.add_argument('--no_autoupdate', action='store_true', help='Disable automatic updates')
        parser.add_argument("--process_name", type=str, help="The name of the PM2 process")
        parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to save/load the checkpoint. If None, the path is set to checkpoint-M<UID>.pth.')
        parser.add_argument('--save-location', type=str, default=None, help='Directory to save/load slice files')
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        config = bt.config(parser)
        if config.test:
            config.subtensor.network = 'test'
            config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'
        elif config.local:
            config.subtensor.network = 'local'
            config.subtensor.chain_endpoint = 'ws://127.0.0.1:9944'
        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()
        if not config.no_autoupdate:
            autoupdater = tplr.AutoUpdate(process_name=config.process_name, bucket_name=config.bucket)
            # Start autoupdater in a new thread
            autoupdate_thread = threading.Thread(target=autoupdater.start)
            autoupdate_thread.daemon = True  # Ensures thread exits when main program exits
            autoupdate_thread.start()
        return config


    def __init__(self):
        # Init config.
        self.config = Miner.config()
        tplr.logger.info('\n' + '-' * 40 + ' Config ' + '-' * 40)
        tplr.logger.info(self.config)

        # Init bittensor objects.
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.chain_manager = tplr.chain.ChainManager(
            subtensor=self.subtensor, wallet=self.wallet, netuid=self.config.netuid
        )
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            tplr.logger.error(f'\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}[/bold]. You need to register first with: [blue]`btcli subnet register`[/blue]\n')
            sys.exit()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        tplr.logger.info('\n' + '-' * 40 + ' Objects ' + '-' * 40)
        tplr.logger.info(f'\nWallet: {self.wallet}\nSubtensor: {self.subtensor}\nMetagraph: {self.metagraph}\nUID: {self.uid}')

        # Init bucket.
        try:
            tplr.logger.info(f'bucket_name: {tplr.config.BUCKET_SECRETS["bucket_name"]}')
            commitment = self.chain_manager.get_commitment(self.uid)
            current_bucket = (
                tplr.config.BUCKET_SECRETS["account_id"] +
                tplr.config.BUCKET_SECRETS["access_key_id"] +
                tplr.config.BUCKET_SECRETS["secret_access_key"]
            )
            if current_bucket != commitment:
                # TODO: Handle mismatched commitments
                raise ValueError("Bucket commitment data does not match.")
            raise ValueError('')
        except Exception:
            tplr.commit(self.subtensor, self.wallet, self.config.netuid)
        tplr.logger.info('Bucket:' + tplr.config.BUCKET_SECRETS["bucket_name"])

        # Init Wandb.
        # Ensure the wandb directory exists
        wandb_dir = os.path.join(os.getcwd(), 'wandb')
        os.makedirs(wandb_dir, exist_ok=True)

        # Define the run ID file path inside the wandb directory
        run_id_file = os.path.join(wandb_dir, f"wandb_run_id_M{self.uid}_{tplr.__version__}.txt")

        # Attempt to read the existing run ID
        if os.path.exists(run_id_file):
            with open(run_id_file, 'r') as f:
                run_id = f.read().strip()
            tplr.logger.info(f"Resuming WandB run with id {run_id}")
        else:
            run_id = None
            tplr.logger.info("Starting a new WandB run.")

        # Initialize WandB
        self.wandb = tplr.initialize_wandb(
            run_prefix='M',
            uid=self.uid,
            config=self.config,
            group='miner',
            job_type='training'
        )

        # Init model.
        tplr.logger.info('\n' + '-' * 40 + ' Hparams ' + '-' * 40)
        self.hparams = tplr.load_hparams()
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        self.model = LlamaForCausalLM(config=self.hparams.model_config)
        self.model.to(self.config.device)
        self.model.train()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,  # Peak learning rate
            betas=(self.hparams.optimizer_beta1, self.hparams.optimizer_beta2),  # B1 and B2
            weight_decay=self.hparams.optimizer_weight_decay,  # Weight decay
            foreach=True,  # more memory usage, but faster
        )   

        # Load checkpoint if it exists
        self.checkpoint_path = f"checkpoint-M{self.uid}.pth" if self.config.checkpoint_path is None else self.config.checkpoint_path 
        if os.path.exists(self.checkpoint_path):
            tplr.logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            global_step, _ = asyncio.run(tplr.load_checkpoint(
                filename=self.checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=None,
                device=self.config.device
            ))

            self.global_step = global_step
            if global_step is None:
                tplr.logger.warning(f"Corrupt checkpoint detected at {self.checkpoint_path}. Removing file and starting fresh.")
                try:
                    os.remove(self.checkpoint_path)
                    tplr.logger.info(f"Removed corrupt checkpoint: {self.checkpoint_path}")
                except OSError as e:
                    tplr.logger.error(f"Failed to remove corrupt checkpoint: {e}")
                global_step = 0
            else:
                tplr.logger.info(f"Resumed from global step {self.global_step}")
        else:
            tplr.logger.info("No checkpoint file found. Starting from scratch.")
            self.global_step = 0

        # Initialize learning rate scheduler
        self.scheduler = tplr.get_wsd_scheduler(
            optimizer=self.optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_stable_steps=self.hparams.num_stable_steps,
            num_decay_steps=self.hparams.num_decay_steps,
        )

        # Init buckets.
        buckets = tplr.get_all_commitments(self.subtensor.substrate, self.config.netuid, self.metagraph)
        self.buckets = []
        buckets = tplr.get_all_commitments(
            substrate=self.subtensor.substrate,
            netuid=self.config.netuid,
            metagraph=self.metagraph
        )

        for uid in self.metagraph.uids:
            bucket = buckets.get(uid)
            tplr.logger.info(f"UID {uid} bucket: {bucket}")

            if bucket is not None:
                tplr.logger.info(f"Retrieved valid bucket for UID {uid}: {bucket.name}")
                self.buckets.append(bucket)
            else:
                tplr.logger.info(f"No valid bucket found for UID {uid}")
                self.buckets.append(None)

        tplr.logger.info(f"Final list of buckets: {self.buckets}")

        # Init run state.
        self.sample_rate = 1.0
        self.current_block = self.subtensor.block
        self.current_window = self.block_to_window( self.current_block )
        self.window_seeds = {self.current_window: self.window_to_seed( self.current_window) }
        self.new_block_event = asyncio.Event()
        self.new_window_event = asyncio.Event()
        self.stop_event = asyncio.Event()    
        self.last_full_steps = self.hparams.desired_batch_size // self.config.actual_batch_size
        bt.logging.off
        self.save_location = self.config.save_location
        if self.save_location is None:
            import tempfile
            self.save_location = tempfile.gettempdir()
        else:
            os.makedirs(self.save_location, exist_ok=True)    
        print ( self.hparams )

    async def update(self):
        """Continuously updates the global state by polling every 10 minutes."""
        await asyncio.sleep(600)  # Initial sleep before starting updates
        while not self.stop_event.is_set():
            st = tplr.T()
            await self.perform_update()
            tplr.logger.info(f"{tplr.P(self.current_window, tplr.T() - st)} Updated global state.")
            await asyncio.sleep(600)

    async def perform_update(self):
        """Updates subtensor connection, metagraph, hyperparameters, and buckets."""
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)

        # Fetch all commitments at once
        buckets = tplr.get_all_commitments(
            substrate=self.subtensor.substrate,
            netuid=self.config.netuid,
            metagraph=self.metagraph
        )

        # Fetch all commitments at once
        buckets = tplr.get_all_commitments(
            substrate=self.subtensor.substrate,
            netuid=self.config.netuid,
            metagraph=self.metagraph
        )

        self.buckets = []
        for uid in self.metagraph.uids:
            bucket = buckets.get(uid)
            if isinstance(bucket, bytes):
                bucket = bucket.decode('utf-8')
            # if bucket is not None and tplr.is_valid_bucket(bucket):
            if bucket is not None:
                tplr.logger.debug(f"UID {uid}: Valid bucket found: {bucket}")
                self.buckets.append(bucket)
            else:
                tplr.logger.debug(f"UID {uid}: Invalid or missing bucket: {bucket}")
                self.buckets.append(None)

    async def run(self):
        # Main loop.
        self.loop = asyncio.get_running_loop()
        self.update_task = asyncio.create_task(self.update())
        self.listener = threading.Thread(target=self.block_listener, args=(self.loop,), daemon=True).start()

        # Optionally sync the model state by pulling model states from the history.
        if self.config.sync_state:
            st = tplr.T()
            history_windows = [self.current_window - i for i in range(self.hparams.max_history - 1, -1, -1)]
            for window in tqdm(history_windows, desc="Syncing state"):
                max_global_step = await tplr.apply_slices_to_model( 
                    model = self.model, 
                    window = window,
                    seed = window,
                    compression = self.hparams.compression,
                    save_location=self.save_location,
                    key = 'state'
                )
                if max_global_step is not None:
                    self.global_step = max(self.global_step, max_global_step)
                    self.scheduler.last_epoch = self.global_step - 1  # Update scheduler
                tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Applied history and updated global step to {self.global_step}.")
            torch.cuda.empty_cache()

        # Main training loop.
        while True:
            try:      
                # Start the window step.     
                tplr.logger.info('[bold]' + '\n' + '-' * 40 + f' Step: {self.global_step} ' + '-' * 40)
                self.global_step += 1

                # Save checkpoint every 500 steps
                if self.global_step % 500 == 0:
                    tplr.logger.info(f"Scheduling checkpoint save at global step {self.global_step}")
                    # Schedule the tplr.save_checkpoint function to run asynchronously
                    asyncio.create_task(tplr.save_checkpoint(
                        filename=self.checkpoint_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        global_step=self.global_step
                    ))
                start_step = tplr.T()
                window = self.current_window

                # Run for non-baseline miners.
                if not self.config.baseline:
                    st = tplr.T()
                    valid_buckets = [b for b in self.buckets if b is not None]

                    if not valid_buckets:
                        tplr.logger.info(f"No valid buckets to download state slices for window {window}")
                        # Wait for the next window
                        while self.current_window == window:
                            await asyncio.sleep(0.1)
                        continue

                    state_slices = await tplr.download_slices_for_buckets_and_windows(
                        buckets=valid_buckets,
                        windows=[window],
                        key='state',
                        save_location=self.save_location
                    )

                    n_state_slices = len(state_slices[window]) if window in state_slices else 0
                    tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Downloaded {n_state_slices} window states.")

                    # Download the delta from the previous window.
                    st = tplr.T()
                    delta_slices = await tplr.download_slices_for_buckets_and_windows(
                        buckets = self.buckets,
                        windows = [ window - 1 ],
                        key = 'delta',
                        save_location=self.save_location
                    )       
                    n_slices = len(delta_slices[ window - 1  ]) if window - 1 in delta_slices else 0
                    tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Download {n_slices} window deltas.")

                    # Apply the state for the current window.
                    st = tplr.T()
                    max_global_step = await tplr.apply_slices_to_model( 
                        model=self.model, 
                        window=window,
                        seed=window,
                        compression=self.hparams.compression,
                        save_location=self.save_location,
                        key='state'
                    )
                    if max_global_step is not None:
                        self.global_step = max(self.global_step, max_global_step)
                        self.scheduler.last_epoch = self.global_step - 1  # Update scheduler
                    tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Applied window state and updated global step to {self.global_step}.")

                # Download the page for the current window.
                st = tplr.T()
                pages = await tplr.dataset.DatasetLoader.next_pages(
                    offset = window,
                    n_pages = self.hparams.validator_window_eval_size,
                    seed = self.uid if not self.config.random else random.randint(0, 1000)
                )
                random.shuffle( pages )
                dataset = await tplr.dataset.DatasetLoader.create(
                    batch_size = self.config.actual_batch_size,
                    sequence_length = self.hparams.sequence_length,
                    pages_info = pages,
                    tokenizer = self.hparams.tokenizer
                )
                tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Downloaded training page: [light_steel_blue]{[p[1] for p in pages]}[/light_steel_blue] random = {self.config.random}")

                # Accumualte gradients on the model applied to the base state.
                train_start = tplr.T()
                self.model.zero_grad()
                self.model.eval()
                total_loss = 0.0
                full_steps = 0
                total_steps = 0
                exhausted_window = False
                for batch in dataset:
                    total_steps += 1
                    if random.random() < self.sample_rate and not exhausted_window:
                        full_steps += 1
                        input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                        labels = input_ids.clone()
                        labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)
                        with torch.amp.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):  # Enable autocasting
                            outputs = self.model(input_ids=input_ids, labels=labels)
                        total_loss += outputs.loss.item()
                        outputs.loss.backward()
                        if window != self.current_window and not self.config.baseline:
                            exhausted_window = True
                            continue
                if self.hparams.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                step_loss = total_loss/(full_steps+1)
                train_duration = tplr.T() - train_start
                tokens_per_step = self.hparams.sequence_length * self.config.actual_batch_size * (full_steps + 1)
                tokens_per_second =  tokens_per_step / train_duration
                tplr.logger.info(f"{tplr.P(window, train_duration)} Accumulated gradients:")
                tplr.logger.info(f"{tplr.P(window, train_duration)} \tTotal steps: [tan]{full_steps}/{total_steps}[/tan], Rate: [tan]{(full_steps/total_steps):.2f}[/tan], Target: [tan]{self.sample_rate:.2f}[/tan]")
                tplr.logger.info(f"{tplr.P(window, train_duration)} \tTotal tokens: [tan]{tokens_per_step}[/tan], Tokens per second: [tan]{tokens_per_second:.2f}[/tan]")
                tplr.logger.info(f"{tplr.P(window, train_duration)} \tLoss: [tan]{step_loss}[tan]")
                if exhausted_window:
                    self.sample_rate = max(0.0001, self.sample_rate * 0.95)
                else:
                    self.sample_rate = min(1, self.sample_rate * 1.05)

                # Run for non-baseline nodes.
                if not self.config.baseline:
                    # Upload the delta for the previous window.
                    st = tplr.T()
                    await tplr.upload_slice_for_window(
                        bucket = tplr.config.BUCKET_SECRETS["bucket_name"],
                        model = self.model, 
                        window = window,
                        seed = window,
                        wallet = self.wallet, 
                        compression = self.hparams.compression,
                        save_location = self.save_location,
                        key = 'delta',
                        global_step = self.global_step 
                    )                
                    tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Uploaded the delta.")

                    # Apply the delta from the previous window.
                    st = tplr.T()
                    max_global_step = await tplr.apply_slices_to_model(
                        model=self.model, 
                        window=window - 1,
                        seed=window - 1,
                        compression=self.hparams.compression,
                        save_location=self.save_location,
                        key='delta'
                    )
                    if max_global_step is not None:
                        self.global_step = max(self.global_step, max_global_step)
                        self.scheduler.last_epoch = self.global_step - 1  # Update scheduler
                    tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Applied window delta and updated global step to {self.global_step}.")

                    # Upload the state for the current window.
                    st = tplr.T()
                    await tplr.upload_slice_for_window(
                        bucket = tplr.config.BUCKET_SECRETS["bucket_name"],
                        model = self.model, 
                        window = window + 1,
                        seed = window + 1, 
                        wallet = self.wallet, 
                        compression = self.hparams.compression,
                        save_location = self.save_location,
                        key = 'state',
                        global_step = self.global_step 
                    )
                    tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Uploaded the state.")

                    # Clean file history.
                    st = tplr.T()
                    await tplr.delete_files_before_window(window_max=window - self.hparams.max_history, save_location=self.save_location, key='state')
                    await tplr.delete_files_before_window(window_max=window - self.hparams.max_history, save_location=self.save_location, key='delta')
                    await tplr.delete_files_from_bucket_before_window( bucket = tplr.config.BUCKET_SECRETS["bucket_name"], window_max = window - self.hparams.max_history, key = 'state' )
                    await tplr.delete_files_from_bucket_before_window( bucket = tplr.config.BUCKET_SECRETS["bucket_name"], window_max = window - self.hparams.max_history, key = 'delta' )
                    tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Cleaned file history.")

                    # Wait until we are on a new window.
                    end_step = tplr.T()
                    while self.current_window == window:
                        await asyncio.sleep(0.1)
                    window_time_delta = self.window_time - end_step
                    window_delta_str = f"[red]{window_time_delta:.2f}[/red]" if window_time_delta < 0 else f"[green]+{window_time_delta:.2f}[/green]"
                    tplr.logger.info(f"{tplr.P(window, end_step - start_step)}[{window_delta_str}]: Finished step.")
                    wandb.log({
                        "miner/loss": step_loss,
                        "miner/tokens_per_step": tokens_per_step,
                        "miner/tokens_per_second": tokens_per_second,
                        "miner/sample_rate": self.sample_rate,
                        "miner/utilization": train_duration / (end_step - start_step),
                        "miner/learning_rate": self.scheduler.get_last_lr()[0]
                    }, step=self.global_step)

            # Catch keyboard interrrupt.
            except KeyboardInterrupt:
                tplr.logger.info("Training interrupted by user. Stopping the run.")
                self.stop_event.set()
                await self.update_task
                sys.exit(0)

            # Catch unknown.
            except Exception as e:
                message = f"Exception during training loop: {escape(str(e))}"
                tplr.logger.exception(message)
                continue

    # Returns the slice window based on a blotplr.
    def block_to_window(self, block: int) -> int:
        return int( block / self.hparams.window_length ) # floor

    # Returns the slice window based on a blotplr.
    def window_to_seed(self, window: int) -> int:
        return str( self.subtensor.get_block_hash( window * self.hparams.window_length ) )

    # A listener thread which posts the block event
    # when the chain announces a new blotplr.
    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event['header']['number'])
            loop.call_soon_threadsafe(self.new_block_event.set)
            if self.block_to_window(self.current_block) != self.current_window:
                self.window_seeds[ self.block_to_window(self.current_block) ] = self.window_to_seed( self.block_to_window(self.current_block) )
                self.current_window = self.block_to_window(self.current_block)
                self.window_duration = tplr.T() - self.window_time if hasattr(self, 'window_time') else 0
                self.window_time = tplr.T()
                loop.call_soon_threadsafe(self.new_window_event.set)
                tplr.logger.info(f"{tplr.P(self.current_window, self.window_duration)} New Window.")
        # Run listener with retry.
        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler)
                break
            except Exception as e:
                tplr.logger.error(f"Failed to subscribe to block headers: {e}.\nRetrying in 1 seconds...")
                time.sleep(1)

if __name__ == "__main__":
    asyncio.run(Miner().run())
