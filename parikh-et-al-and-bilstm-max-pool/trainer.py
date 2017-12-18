class Trainer:
    def __init__(self, params, train_data, dev_data,
                 dev_mismatched_data, embedding):
        self.params = params
        self.train_data = train_data
        self.dev_data = dev_data
        self.dev_mismatched_data = dev_mismatched_data
        self.epochs = params.epochs
        print("Creating dataloaders")
        self.cuda_available = torch.cuda.is_available()

        self.train_loader = DataLoader(dataset=train_data,
                                       shuffle=True,
                                       batch_size=params.batch_size,
                                       pin_memory=self.cuda_available,
                                       collate_fn=utils.collate_batch)
        self.dev_loader = DataLoader(dataset=dev_data,
                                     shuffle=False,
                                     batch_size=params.batch_size,
                                     pin_memory=self.cuda_available)
        self.dev_mismatched_loader = DataLoader(dataset=dev_mismatched_data,
                                                shuffle=False,
                                                batch_size=params.batch_size,
                                                pin_memory=self.cuda_available)

        self.string_fixer = "=========="
        self.embedding = embedding
        self.writer = SummaryWriter("/scratch/as10656/nli_models/logs/opti")
        self.writer_step = 0

    def load(self, model_name="decomposable"):
        print("Loading model")

        if model_name == "decomposable":
            self.model = DecomposableAttention(self.params, self.embedding)
        else:
            self.model = BiLSTMMaxPooling(self.params, self.embedding)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                           self.model.parameters()),
                                    lr=params.lr)

        self.start_time = time.time()
        self.histories = {
            "train_loss": np.empty(0, dtype=np.float32),
            "train_acc": np.empty(0, dtype=np.float32),
            "dev_matched_loss": np.empty(0, dtype=np.float32),
            "dev_matched_acc": np.empty(0, dtype=np.float32),
            "dev_mismatched_loss": np.empty(0, dtype=np.float32),
            "dev_mismatched_acc": np.empty(0, dtype=np.float32)
        }

        self.early_stopping = EarlyStopping(
            self.model, self.optimizer, patience=self.params.patience,
            minimize=False)
        if self.params.resume:
            checkpoint = utils.load_checkpoint(self.params.resume)
            if checkpoint is not None:
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.histories.update(checkpoint)
                self.early_stopping.init_from_checkpoint(checkpoint)
                print("Loaded model, Best Loss: %.8f, Best Acc: %.2f" %
                      (checkpoint['best'], checkpoint['best_acc']))

        if self.cuda_available:
            self.model = self.model.cuda()

        print("Model loaded")

    def train(self):
        criterion = nn.CrossEntropyLoss()

        start_epoch = 0
        best_prec = 0

        is_best = False

        self.model.train()
        print("Starting training")
        self.print_info()
        for epoch in range(start_epoch, self.params.epochs):
            for i, (premise, hypo, labels) in enumerate(self.train_loader):
                premise_batch = Variable(premise.long())
                hypo_batch = Variable(hypo.long())
                labels_batch = Variable(labels)
                if self.cuda_available:
                    premise_batch = premise_batch.cuda()
                    hypo_batch = hypo_batch.cuda()
                    labels_batch = labels_batch.cuda(async=True)

                self.optimizer.zero_grad()
                output = self.model(premise_batch, hypo_batch)
                loss = criterion(output, labels_batch.long())
                loss.backward()
                self.optimizer.step()
                if self.params.extra_debug and \
                        (i + 1) % (self.params.batch_size * 4) == 0:
                    for n, p in filter(lambda np: np[1].grad is not None,
                                       self.model.named_parameters()):
                        self.writer.add_histogram(
                            n, p.grad.data.cpu().numpy(),
                            global_step=self.writer_step)

                    self.writer_step += 1

                    print(('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4},')
                          .format(epoch + 1,
                                  self.params.epochs,
                                  i + 1,
                                  len(self.train_loader),
                                  loss.data[0]))
            train_acc, train_loss = self.validate_model(self.train_loader,
                                                        self.model)
            dev_matched_acc, dev_matched_loss = self.validate_model(
                self.dev_loader, self.model)
            dev_mismatched_acc, dev_mismatched_loss = self.validate_model(
                self.dev_mismatched_loader, self.model)

            self.histories['train_loss'] = np.append(
                self.histories['train_loss'],
                [train_loss])
            self.histories['train_acc'] = np.append(
                self.histories['train_acc'],
                [train_acc])
            self.histories['dev_matched_loss'] = np.append(
                self.histories['dev_matched_loss'], [dev_matched_loss])
            self.histories['dev_matched_acc'] = np.append(
                self.histories['dev_matched_acc'], [dev_matched_acc])
            self.histories['dev_mismatched_loss'] = np.append(
                self.histories['dev_mismatched_loss'], [dev_mismatched_loss])
            self.histories['dev_mismatched_acc'] = np.append(
                self.histories['dev_mismatched_acc'], [dev_mismatched_acc])

            if not self.early_stopping(dev_matched_loss, dev_matched_acc,
                                       epoch. self.histories):
                self.print_train_info(epoch, train_acc, train_loss,
                                      dev_matched_acc, dev_matched_loss,
                                      dev_mismatched_acc,
                                      dev_mismatched_loss)
            else:
                print("Early stopping activated")
                print("Restoring earlier state and stopping")
                self.early_stopping.print_info()
                plot_learning_curves(self.histories)
                plt.show()
                break

    def validate_model(self, loader, model):
        model.eval()
        correct = 0
        total = 0
        total_loss = 0

        for premise, hypo, labels in loader:
            premise_batch = Variable(premise.long(), volatile=True)
            hypo_batch = Variable(hypo.long(), volatile=True)
            labels_batch = Variable(labels.long())

            if self.cuda_available:
                premise_batch = premise_batch.cuda()
                hypo_batch = hypo_batch.cuda()
                labels_batch = labels_batch.cuda()

            output = model(premise_batch, hypo_batch)
            loss = nn.functional.cross_entropy(output, labels_batch.long(),
                                               size_average=False)
            total_loss += loss.data[0]
            total += len(labels_batch)

            if not self.cuda_available:
                correct += (labels_batch ==
                            output.max(1)[1]).data.cpu().numpy().sum()
            else:
                correct += (labels_batch == output.max(1)[1]).data.sum()

        model.train()

        average_loss = total_loss / total
        return correct / total * 100, average_loss

    def print_info(self):
        print(self.string_fixer + " Data " + self.string_fixer)
        print("Training set: %d examples" % (len(self.train_data)))
        print("Validation set: %d examples" % (len(self.dev_data)))
        print("Timestamp: %s" % utils.get_time_hhmmss())

        print(self.string_fixer + " Params " + self.string_fixer)

        print("Learning Rate: %f" % self.params.lr)
        print("Dropout (p): %f" % self.params.dropout)
        print("Batch Size: %d" % self.params.batch_size)
        print("Epochs: %d" % self.params.epochs)
        print("Patience: %d" % self.params.patience)
        print("Resume: %s" % self.params.resume)
        print("GRU Encode: %s" % str(self.params.gru_encode))
        print("Cuda: %s" % str(torch.cuda.is_available()))
        print("Batch Optimizations: %s" % str(self.params.use_optimizations))
        print("Intra Attention: %s" % str(self.params.use_intra_attention))
        print("Model Structure:")
        print(self.model)

    def print_train_info(self, epoch, train_acc, train_loss,
                         dev_acc, dev_loss,
                         dev_mismatched_acc, dev_mismatched_loss):
        print((self.string_fixer + " Epoch: {0}/{1} " + self.string_fixer)
              .format(epoch + 1, self.params.epochs))
        print("Train Loss: %.8f, Train Acc: %.2f" % (train_loss, train_acc))
        print("Dev Matched Loss: %.8f, Dev Matched Acc: %.2f" %
              (dev_loss, dev_acc))
        print("Dev Mismatched Loss: %.8f, Dev Mismatched Acc: %.2f" %
              (dev_mismatched_loss, dev_mismatched_acc))
        self.early_stopping.print_info()
        print("Elapsed Time: %s" % (utils.get_time_hhmmss(self.start_time)))
        print("Current timestamp: %s" % (utils.get_time_hhmmss()))
