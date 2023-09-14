import argparse
import sys

# ROOT='X:/Users/localadmin/OneDrive - Vrije Universiteit Brussel/Codes/mca_bert_Soha_01'
ROOT='/home/mahdi/codes/mca_bert_Soha_01'
def args_parser():
    parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")
    parser.add_argument("--data_path", type=str, default=ROOT + '/data/data_giannis/data/mimic3/train_full.csv',
                        help="path to a file containing sorted train data. dev/test splits assumed to have same name format with 'train' replaced by 'dev' and 'test'")
    parser.add_argument("--vocab", type=str, default= ROOT + '/data/data_giannis/data/mimic3/vocab.csv',
                        help="path to a file holding vocab word list for discretizing words")
    parser.add_argument("--Y", type=str, default='full', help="size of label space")
    parser.add_argument("--model", type=str, default='conv_attn',
                        choices=["cnn_vanilla", "rnn", "conv_attn", "multi_conv_attn", "logreg", "saved"], help="model")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs to train")
    parser.add_argument("--embed-file", type=str, default=ROOT + '/data/data_giannis/data/mimic3/processed_full.embed',
                         dest="embed_file",
                        help="path to a file holding pre-trained embeddings")
    parser.add_argument("--cell-type", type=str, choices=["lstm", "gru"], help="what kind of RNN to use (default: GRU)",
                        dest='cell_type',
                        default='gru')
    parser.add_argument("--rnn-dim", type=int,  dest="rnn_dim", default=128,
                        help="size of rnn hidden layer (default: 128)")
    parser.add_argument("--bidirectional", dest="bidirectional", action="store_const",  const=True,
                        help="optional flag for rnn to use a bidirectional model")
    parser.add_argument("--rnn-layers", type=int,  dest="rnn_layers", default=1,
                        help="number of layers for RNN models (default: 1)")
    parser.add_argument("--embed-size", type=int,  dest="embed_size", default=100,
                        help="size of embedding dimension. (default: 100)")
    parser.add_argument("--filter-size", type=str,  dest="filter_size", default=10,
                        help="size of convolution filter to use. (default: 3) For multi_conv_attn, give comma separated integers, e.g. 3,4,5")
    parser.add_argument("--num-filter-maps", type=int,  dest="num_filter_maps", default=50,
                        help="size of conv output (default: 50)")
    parser.add_argument("--pool", choices=['max', 'avg'],  dest="pool",
                        help="which type of pooling to do (logreg model only)")
    parser.add_argument("--code-emb", type=str,  dest="code_emb",
                        help="point to code embeddings to use for parameter initialization, if applicable")
    parser.add_argument("--weight-decay", type=float,  dest="weight_decay", default=0,
                        help="coefficient for penalizing l2 norm of model weights (default: 0)")
    parser.add_argument("--lr", type=float,  dest="lr", default=1e-3,
                        help="learning rate for Adam optimizer (default=1e-3)")
    parser.add_argument("--batch-size", type=int,  dest="batch_size", default=16,
                        help="size of training batches")
    parser.add_argument("--dropout", dest="dropout", type=float,  default=0.2,
                        help="optional specification of dropout (default: 0.5)")
    parser.add_argument("--lmbda", type=float,  dest="lmbda", default=0,
                        help="hyperparameter to tradeoff BCE loss and similarity embedding loss. defaults to 0, which won't create/use the description embedding module at all. ")
    parser.add_argument("--dataset", type=str, choices=['mimic2', 'mimic3'], dest="version", default='mimic3',

                        help="version of MIMIC in use (default: mimic3)")
    parser.add_argument("--test-model", type=str,dest="test_model",# default='/home/soha/codes/caml_transfer_icd/saved_models/conv_attn_Feb_22_09_34_21/model_best_f1_macro.pth',
                        help="path to a saved model to load and evaluate")
    parser.add_argument("--criterion", type=str, default='f1_micro',  dest="criterion",
                        help="which metric to use for early stopping (defaultc: f1_micro)")
    parser.add_argument("--patience", type=int, default=10,
                        help="how many epochs to wait for improved criterion metric before early stopping (default: 3)")
    parser.add_argument("--gpu", default=True, dest="gpu", action="store_const",  const=True,
                        help="optional flag to use GPU if available")
    parser.add_argument("--public-model", dest="public_model", action="store_const",  const=True,
                        help="optional flag for testing pre-trained models from the public github")
    parser.add_argument("--stack-filters", dest="stack_filters", action="store_const",  const=True,
                        help="optional flag for multi_conv_attn to instead use concatenated filter outputs, rather than pooling over them")
    parser.add_argument("--samples", dest="samples", action="store_const",  const=True,
                        help="optional flag to save samples of good / bad predictions")
    parser.add_argument("--quiet", dest="quiet", action="store_const",  const=True,
                        help="optional flag not to print so much during training")
    ####### BY SOHA
    parser.add_argument("--ontology", type=bool, default=True, help="add external knowledge to the model (subcode has to be true)")
    parser.add_argument("--transfer", type=bool, default=False, help="is this part of the transfer learning experiment?")
    parser.add_argument("--transfer_10", type=bool, default=False, help="if part of transfer learnining, is it for icd 10?")
    parser.add_argument("--subcode", type=bool, default=True, help="to train with subset of mimic3 codes that are compatible with ICD 10 and icdcodex")
    parser.add_argument("--from_checkpoint", type=bool, default=False,
                        help="continue learning from checkpoint?")
    parser.add_argument("--checkpoint", default='/home/soha/codes/caml_transfer_icd/saved_models/conv_attn_Jan_30_15_28_47/model_best_f1_micro.pth', help="path to checkpoint")
    #'/mnt/data2/soha/codes/caml_transfer_icd/saved_models/conv_attn_Jan_17_14_33_29/model_best_f1_micro.pth', help="path to checkpoint")
    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    return args

# import argparse
# import sys
#
# ROOT='C:/Users/localadmin/OneDrive - Vrije Universiteit Brussel/Codes/mca_bert_Soha_01'
# # ROOT='/home/soha/codes/mca_bert_Soha_01'
# def args_parser():
#     parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")
#     parser.add_argument("data_path", type=str, default=ROOT + '/data/data_giannis/data/mimic3/train_full.csv',
#                         help="path to a file containing sorted train data. dev/test splits assumed to have same name format with 'train' replaced by 'dev' and 'test'")
#     parser.add_argument("vocab", type=str, default= ROOT + '/data/data_giannis/data/mimic3/vocab.csv',
#                         help="path to a file holding vocab word list for discretizing words")
#     parser.add_argument("Y", type=str, default='full', help="size of label space")
#     parser.add_argument("model", type=str, default='conv_attn',
#                         choices=["cnn_vanilla", "rnn", "conv_attn", "multi_conv_attn", "logreg", "saved"], help="model")
#     parser.add_argument("n_epochs", type=int, default=200, help="number of epochs to train")
#     parser.add_argument("--embed-file", type=str, default=ROOT + '/data/data_giannis/data/mimic3/processed_full.embed',
#                         required=False, dest="embed_file",
#                         help="path to a file holding pre-trained embeddings")
#     parser.add_argument("--cell-type", type=str, choices=["lstm", "gru"], help="what kind of RNN to use (default: GRU)",
#                         dest='cell_type',
#                         default='gru')
#     parser.add_argument("--rnn-dim", type=int, required=False, dest="rnn_dim", default=128,
#                         help="size of rnn hidden layer (default: 128)")
#     parser.add_argument("--bidirectional", dest="bidirectional", action="store_const", required=False, const=True,
#                         help="optional flag for rnn to use a bidirectional model")
#     parser.add_argument("--rnn-layers", type=int, required=False, dest="rnn_layers", default=1,
#                         help="number of layers for RNN models (default: 1)")
#     parser.add_argument("--embed-size", type=int, required=False, dest="embed_size", default=100,
#                         help="size of embedding dimension. (default: 100)")
#     parser.add_argument("--filter-size", type=str, required=False, dest="filter_size", default=10,
#                         help="size of convolution filter to use. (default: 3) For multi_conv_attn, give comma separated integers, e.g. 3,4,5")
#     parser.add_argument("--num-filter-maps", type=int, required=False, dest="num_filter_maps", default=50,
#                         help="size of conv output (default: 50)")
#     parser.add_argument("--pool", choices=['max', 'avg'], required=False, dest="pool",
#                         help="which type of pooling to do (logreg model only)")
#     parser.add_argument("--code-emb", type=str, required=False, dest="code_emb",
#                         help="point to code embeddings to use for parameter initialization, if applicable")
#     parser.add_argument("--weight-decay", type=float, required=False, dest="weight_decay", default=0,
#                         help="coefficient for penalizing l2 norm of model weights (default: 0)")
#     parser.add_argument("--lr", type=float, required=False, dest="lr", default=1e-3,
#                         help="learning rate for Adam optimizer (default=1e-3)")
#     parser.add_argument("--batch-size", type=int, required=False, dest="batch_size", default=16,
#                         help="size of training batches")
#     parser.add_argument("--dropout", dest="dropout", type=float, required=False, default=0.2,
#                         help="optional specification of dropout (default: 0.5)")
#     parser.add_argument("--lmbda", type=float, required=False, dest="lmbda", default=0,
#                         help="hyperparameter to tradeoff BCE loss and similarity embedding loss. defaults to 0, which won't create/use the description embedding module at all. ")
#     parser.add_argument("--dataset", type=str, choices=['mimic2', 'mimic3'], dest="version", default='mimic3',
#                         required=False,
#                         help="version of MIMIC in use (default: mimic3)")
#     parser.add_argument("--test-model", type=str, dest="test_model", required=False,
#                         help="path to a saved model to load and evaluate")
#     parser.add_argument("--criterion", type=str, default='prec_at_8', required=False, dest="criterion",
#                         help="which metric to use for early stopping (default: f1_micro)")
#     parser.add_argument("--patience", type=int, default=10, required=False,
#                         help="how many epochs to wait for improved criterion metric before early stopping (default: 3)")
#     parser.add_argument("--gpu", default=True, dest="gpu", action="store_const", required=False, const=True,
#                         help="optional flag to use GPU if available")
#     parser.add_argument("--public-model", dest="public_model", action="store_const", required=False, const=True,
#                         help="optional flag for testing pre-trained models from the public github")
#     parser.add_argument("--stack-filters", dest="stack_filters", action="store_const", required=False, const=True,
#                         help="optional flag for multi_conv_attn to instead use concatenated filter outputs, rather than pooling over them")
#     parser.add_argument("--samples", dest="samples", action="store_const", required=False, const=True,
#                         help="optional flag to save samples of good / bad predictions")
#     parser.add_argument("--quiet", dest="quiet", action="store_const", required=False, const=True,
#                         help="optional flag not to print so much during training")
#     args = parser.parse_args()
#     command = ' '.join(['python'] + sys.argv)
#     args.command = command
#     return args
