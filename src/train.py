import seaborn as sns
class Trainer:
    def __init__(self,
                 net: nn.Module,
                 dataset: torch.utils.data.Dataset,
                 criterion: nn.Module,
                 lr: float,
                 accumulation_steps: int,
                 batch_size: int,
                 fold: int,
                 num_epochs: int,
                 path_to_csv: str,
                 display_plot: bool = True,
                ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device:", self.device)
        self.display_plot = display_plot
        self.net = net.to(self.device)
        self.criterion = criterion
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=4, verbose=True)
        self.accumulation_steps = accumulation_steps // batch_size
        self.phases = ["train", "val"]
        self.num_epochs = num_epochs

        # Initialize fresh dataloaders
        self.dataloaders = {
            phase: get_dataloader(
                dataset=dataset,
                path_to_csv=path_to_csv,
                phase=phase,
                fold=fold,
                batch_size=batch_size,
                num_workers=4,
                #augment_prob= 0.1,  # Probability of applying augmentations
            )
            for phase in self.phases
        }

        # Ensure fresh training: Reset best loss and training logs
        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.jaccard_scores = {phase: [] for phase in self.phases}
        self.sen_scores = {phase: [] for phase in self.phases}
        self.spf_scores = {phase: [] for phase in self.phases}
        self.acc_scores = {phase: [] for phase in self.phases}

    def _compute_loss_and_outputs(self, images: torch.Tensor, targets: torch.Tensor):
        images, targets = images.to(self.device), targets.to(self.device)
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits

    def _do_epoch(self, epoch: int, phase: str):
        print(f"{phase} epoch: {epoch} | time: {time.strftime('%H:%M:%S')}")
        self.net.train() if phase == "train" else self.net.eval()
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()

       # Progress bar using tqdm
        with tqdm(total=total_batches, desc=f"{phase.upper()} {epoch+1}", unit="batch") as pbar:
            for itr, data_batch in enumerate(dataloader):
                images, targets = data_batch['image'], data_batch['mask']
                loss, logits = self._compute_loss_and_outputs(images, targets)
                loss = loss / self.accumulation_steps

                if phase == "train":
                    loss.backward()
                    if (itr + 1) % self.accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                running_loss += loss.item()
                meter.update(logits.detach().cpu(), targets.detach().cpu())

                # Update progress bar with loss
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        epoch_dice, epoch_iou, epoch_sen, epoch_spf, epoch_acc = meter.get_metrics()

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.jaccard_scores[phase].append(epoch_iou)
        self.sen_scores[phase].append(epoch_sen)
        self.spf_scores[phase].append(epoch_spf)
        self.acc_scores[phase].append(epoch_acc)

        return epoch_loss

    def run(self):
        print("Starting training from scratch...")

        for epoch in range(self.num_epochs):
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "val")
                self.scheduler.step(val_loss)

            # Save best model
            if val_loss < self.best_loss:
                print(f"\n{'#' * 20}\nSaved new checkpoint\n{'#' * 20}\n")
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), "best_model.pth")

            print()
        
        if self.display_plot:
            self._plot_train_history()

        self._save_train_history()

    def _plot_train_history(self):
        data = [self.losses, self.dice_scores, self.jaccard_scores, self.acc_scores, self.sen_scores, self.spf_scores]
        colors = ['deepskyblue', "crimson"]
        labels = [
            f"""
            train loss {self.losses['train'][-1]}
            val loss {self.losses['val'][-1]}
            """,

            f"""
            train dice score {self.dice_scores['train'][-1]}
            val dice score {self.dice_scores['val'][-1]}
            """,

            f"""
            train jaccard score {self.jaccard_scores['train'][-1]}
            val jaccard score {self.jaccard_scores['val'][-1]}
            """,

            f"""
            train acc score {self.acc_scores['train'][-1]}
            val acc score {self.acc_scores['val'][-1]}
            """,

            f"""
            train sen score {self.sen_scores['train'][-1]}
            val sen score {self.sen_scores['val'][-1]}
            """,

            f"""
            train spf score {self.spf_scores['train'][-1]}
            val spf score {self.spf_scores['val'][-1]}
            """,
        ]

        with plt.style.context("seaborn-dark-palette"):
        # Dynamically calculate the number of rows and columns
            num_plots = len(data)
            num_cols = 3  # Number of columns
            num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate rows dynamically

            # Create subplots
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 10 * num_rows))
            axes = axes.flatten()  # Flatten the 2D array into 1D

            # Plot each metric
            for i in range(num_plots):
                
                ax = axes[i]
                ax.plot(data[i]['val'], c=colors[0], label="val")
                ax.plot(data[i]['train'], c=colors[-1], label="train")
                ax.set_title(labels[i])
                ax.legend(loc="upper right")

            # Hide unused subplots
            for i in range(num_plots, num_rows * num_cols):
               
                axes[i].axis('off')

            plt.tight_layout()
            plt.show()


    def _save_train_history(self):
        print("Saving final model and logs...")
        torch.save(self.net.state_dict(), "final_model.pth")

        logs_ = [self.losses, self.dice_scores, self.jaccard_scores, self.acc_scores, self.sen_scores, self.spf_scores]
        log_names_ = ["_loss", "_dice", "_jaccard","_acc", "_sen", "_spf"]
        logs = [logs_[i][key] for i in range(len(logs_)) for key in logs_[i]]
        log_names = [key + log_names_[i] for i in range(len(logs_)) for key in logs_[i]]

        pd.DataFrame(dict(zip(log_names, logs))).to_csv("train_log.csv", index=False)
