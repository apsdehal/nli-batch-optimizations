num_words = 100140
batch_size = 32
valid_batch_size = 32
new_batch_size = 11

dim_word = 300
char_nout = 100
dim_char_emb = 15
learning_rate = 0.0004
dim_hidden = 600
maxlen = 100
reload = False
batch_processing = False
num_epochs = 10
eval_period = 2000
waiting_period = 1
patience = 5

const_last_exec = 'uidx'
const_model_state_dict = 'state_dict'
const_best_val_acc = 'best_val'
const_optimizer_state_dict = 'optimizer'
const_train_acc_history = 'train_hist'
const_valid_acc_history = 'valid_hist'
const_valid_out_domain_acc_history = 'valid_out_domain_hist'
const_train_loss_history = 'train_loss_hist'
const_valid_loss_history = 'valid_loss_hist'
const_valid_out_domain_loss_history = 'valid_out_domain_loss_hist'

const_wait_counter = 'wait_counter'
const_epoch = 'epoch'
train_file_loc = "data/multinli_1.0/multinli_1.0_train.txt"
dev_file_loc = "data/multinli_1.0/multinli_1.0_dev_matched.txt"
valid_out_domain_file_loc = "data/multinli_1.0/multinli_1.0_dev_mismatched.txt"

log_file_name = 'NormallyProcessedLog'
model_path = 'NormallyProcessedNLIModel'

print_model_load = '----Model Loaded-----'
print_no_reload = '----Creating Params-----'
print_optimizer = '----Optimizer Created-----'
print_reload = '----Reloading Params-----'
print_training = '-----Training Started-----'
