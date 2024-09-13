import os
import numpy as np
import collections
import matplotlib
import tqdm
from IPython.core.display import display, HTML
from methods.bag_of_ngrams.processing import cleanReports, cleanSplit, stripChars
from pyfunctions.config import BASE_DIR
from pyfunctions.general import extractListFromDic, readJson
from pyfunctions.pathology import extract_synoptic, fixLabel, exclude_labels
from pyfunctions.cdt_basic import comp_cd_scores_level_skip, get_encoding
from sklearn import preprocessing
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn.functional as F

import shap
import scipy as sp
import lime
from lime.lime_text import LimeTextExplainer, IndexedString, TextDomainMapper
from pyfunctions._integrated_gradients import get_input_data, ig_attribute
from captum.attr import (IntegratedGradients)

############ MAIN FUNCTIONS TO CALL ############
def load_data_and_model(data_name, model_type, device):
    identifier = f'{data_name}_{model_type}'
    if identifier == "pathology_bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        data_path = os.path.join(BASE_DIR, "data/prostate.json")
        data, le_dict = load_path_data(data_path, tokenizer)
        # load model
        model_path = os.path.join(BASE_DIR, "models/path/bert_PrimaryGleason")
        model_checkpoint_file = os.path.join(model_path, "save_output")
        model = load_path_model(model_checkpoint_file, le_dict)
    elif identifier == "pathology_pubmed_bert":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        data_path = os.path.join(BASE_DIR, "data/prostate.json")
        data, le_dict = load_path_data(data_path, tokenizer)
        # load model
        model_path = os.path.join(BASE_DIR, "models/path/pubmed_bert_PrimaryGleason")
        model_checkpoint_file = os.path.join(model_path, "save_output")
        model = load_path_model(model_checkpoint_file, le_dict)
    elif identifier == "sst2_bert":
        tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
        data, le_dict = load_sst2_data()
        # load model
        model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
    elif identifier == "agnews_bert":
        tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
        data, le_dict = load_agnews_data()
        # load model
        model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
        
    model = model.eval()
    model.to(device)
    return data, le_dict, tokenizer, model


def run_local_importance(text, label, model, tokenizer, le_dict, device, max_seq_len, method, class_names, IG_interpretable_embeds, level=0, skip=1, num_at_time=64):
    #tokens = tokenizer.convert_ids_to_tokens(tokenizer(text)["input_ids"])
    #intervals, words = compute_word_intervals(tokens)
    encoding = get_encoding(text, tokenizer, device, max_seq_len=max_seq_len)
    toks = tokenizer.convert_ids_to_tokens([x for x in encoding['input_ids'][0] if x !=0 ])
    intervals, words = compute_word_intervals(toks)
    with torch.no_grad():
        if method == "CDT":
            scores, irrel_scores = comp_cd_scores_level_skip(model, encoding, label, le_dict, device, max_seq_len=max_seq_len, level=level, skip=skip, num_at_time=num_at_time)
            visualize_cdt(scores, irrel_scores, intervals, words)
        elif method == "lime":
            # LIME has its own tokenizing scheme - split by spaces
            scores, words = run_lime(text, class_names, model, tokenizer, device, label, le_dict, max_seq_len=max_seq_len)
            visualize_common(scores, words, method)
        elif method == "shap":
            scores = run_shap(text, model, tokenizer, intervals, device, label, le_dict, max_seq_len=max_seq_len)
            visualize_common(scores, words, method)
        elif method == "IG":
            scores = run_ig(text, model, tokenizer, intervals, device, label, le_dict, IG_interpretable_embeds, max_seq_len=max_seq_len)
            visualize_common(scores, words, method)
    return scores
#################################################
# IG
def run_ig(text, model, tokenizer, intervals, device, label, le_dict, IG_interpretable_embeds, max_seq_len):
    def predict_forward_func(input_ids, token_type_ids=None,
                         position_ids=None, attention_mask=None):
        """Function passed to ig constructors"""
        return model(inputs_embeds=input_ids,
                     token_type_ids=token_type_ids,
                     position_ids=position_ids,
                     attention_mask=attention_mask)[0]
    
    ig = IntegratedGradients(predict_forward_func)
    interpretable_embedding1, interpretable_embedding2, interpretable_embedding3 = IG_interpretable_embeds
    #model.to(device)
    input_data, input_data_embed = get_input_data(interpretable_embedding1, interpretable_embedding2, interpretable_embedding3,
                                                  text, tokenizer, max_seq_len, device)

    attributions, approximation_error = ig_attribute(ig, int(le_dict[label]), input_data_embed)
    scores = attributions[0].detach().cpu().numpy().squeeze().sum(1)

    #tokens = tokenizer.convert_ids_to_tokens(tokenizer(text)["input_ids"])
    #intervals, words = compute_word_intervals(tokens)
    word_scores = combine_token_scores(intervals, scores)

    return word_scores

# SHAP
def run_shap(text, model, tokenizer, intervals, device, label, le_dict, max_seq_len):
    def shap_predictor(texts):
        tv = torch.tensor(
            [
                tokenizer.encode(v, padding="max_length", max_length=max_seq_len, truncation=True) for v in texts
            ]
        ).to(device)

        outputs = model(tv)[0].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores[:, le_dict[label]])
        return val

    explainer = shap.Explainer(shap_predictor, tokenizer)
    scores = explainer([text], fixed_context=1)
    word_scores = combine_token_scores(intervals, scores.values[0])
    return word_scores

# LIME
def run_lime(text, class_names, model, tokenizer, device, label, le_dict, max_seq_len):
    def lime_predictor(texts):
        batch_size = 128
        if len(texts) % batch_size == 0:
            max_epochs = len(texts) // batch_size
        else:
            max_epochs = len(texts) // batch_size + 1

        total_probas = []
        for L in tqdm.tqdm(range(max_epochs), desc="LIME Batch Processing..."):
            start = batch_size*L
            end = batch_size*(L+1) if len(texts) > batch_size*(L+1) else len(texts)
            outputs = model(**tokenizer(texts[start:end], add_special_tokens=True, 
                             max_length=max_seq_len,
                             truncation=True, 
                             padding = "max_length", 
                             return_attention_mask=True, 
                             pad_to_max_length=True, return_tensors="pt").to(device))
            tensor_logits = outputs[0]
            probas = F.softmax(tensor_logits).detach().cpu().numpy()
            total_probas.extend(probas)
        total_probas = np.stack(total_probas) #[num_samples, num_classes]
        return total_probas

    explainer = LimeTextExplainer(class_names=class_names, bow=False, split_expression=' ')
    indexed_text = IndexedString(text, bow=False, split_expression=' ')
    vocab_size = indexed_text.num_words()
    exp = explainer.explain_instance(text, lime_predictor, num_features=vocab_size, labels=[le_dict[label]])

    scores = exp.local_exp[le_dict[label]]
    mapper = TextDomainMapper(indexed_text)
    combine_to_weight = mapper.map_exp_ids(scores, positions=True) #[(word_pos, weight)]
    
    pos2word, pos2weight = {}, {}
    for combine, w in combine_to_weight:
        tag = combine.find('_')
        word = combine[:tag]
        pos = int(combine[tag+1:])
        pos2word[pos] = word
        pos2weight[pos] = w

    od = collections.OrderedDict(sorted(pos2word.items()))
    reconstructed_s = ' '.join([item[1] for item in od.items()])
    assert(reconstructed_s == text)
    
    scores = [pos2weight[pos] for pos in od.keys()]
    words = [pos2word[pos] for pos in od.keys()]

    return scores, words

def visualize_common(word_scores, words, method):
    assert(len(word_scores) == len(words))
    normalized = normalize_word_scores(word_scores)
    print(f'Viz {method}: ')
    display_colored_html(words, normalized)

# visualization helper
def visualize_cdt(scores, irrel_scores, intervals, words):
    #toks = tokenizer.convert_ids_to_tokens([x for x in encoding['input_ids'][0] if x !=0 ])
    #intervals, words = compute_word_intervals(toks)

    word_scores = combine_token_scores(intervals, scores)
    irrel_word_scores = combine_token_scores(intervals, irrel_scores) # for ablation purpose
    
    normalized = normalize_word_scores(word_scores)
    irrel_normalized = normalize_word_scores(irrel_word_scores)
    
    print("Viz rel: ")
    display_colored_html(words, normalized)
    print("Viz irrel: ")
    display_colored_html(words, irrel_normalized)
    print("Viz rel-irrel: ")
    display_colored_html(words, normalized - irrel_normalized)
    
def display_colored_html(words, scores):
    s = colorize(words, scores)
    display(HTML(s))

def normalize_word_scores(word_scores):
    neg_pos_lst = [i for i, x in enumerate(word_scores) if x < 0]
    abs_word_scores = np.abs(word_scores)
    normalized = (abs_word_scores-min(abs_word_scores))/(max(abs_word_scores)-min(abs_word_scores)) # in [0, 1] range
    for i, x in enumerate(normalized):
        if i in neg_pos_lst:
            normalized[i] = -normalized[i]
    return normalized
            
def chop_cmap_frac(cmap: LinearSegmentedColormap, frac: float) -> LinearSegmentedColormap:
    """Chops off the ending 1- `frac` fraction of a colormap."""
    cmap_as_array = cmap(np.arange(256))
    cmap_as_array = cmap_as_array[:int(frac * len(cmap_as_array))]
    return LinearSegmentedColormap.from_list(cmap.name + f"_frac{frac}", cmap_as_array)

def colorize(words, color_array, mid=0):
    cmap_pos = LinearSegmentedColormap.from_list('', ['white', '#48b6df'])
    cmap_neg = LinearSegmentedColormap.from_list('', ['white', '#dd735b'])
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        if color > mid:
          color = matplotlib.colors.rgb2hex(cmap_pos(color)[:3])
        elif color < mid:
          color = matplotlib.colors.rgb2hex(cmap_neg(abs(color))[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string

def compute_word_intervals(token_lst):
    word_cnt = 0
    interval_dict = collections.defaultdict(list)

    pretok_sent = ""

    tokens_len = len(token_lst)
    for i in range(tokens_len):
        tok = token_lst[i]
        if tok.startswith("##"):
            interval_dict[word_cnt].append(i)
            pretok_sent += tok[2:]
        else:
            word_cnt += 1
            interval_dict[word_cnt].append(i)
            pretok_sent += " " + tok
    pretok_sent = pretok_sent[1:]
    word_lst = pretok_sent.split(" ")

    assert(len(interval_dict) == len(word_lst))

    return interval_dict, word_lst

def combine_token_scores(interval_dict, scores):
    word_cnt = len(interval_dict)
    new_scores = np.zeros(word_cnt)
    for i in range(word_cnt):
        t_idx_lst = interval_dict[i+1]
        if len(t_idx_lst) == 1:
            new_scores[i] = scores[t_idx_lst[0]]
        else:
            new_scores[i] = np.sum(scores[t_idx_lst[0]:t_idx_lst[-1]+1])
    return new_scores

# data helper
def load_agnews_data():
    raw_agnews = load_dataset('ag_news', split='test')
    label_classes = np.unique(raw_agnews['label'])
    le = preprocessing.LabelEncoder()
    le.fit(label_classes)

    # Map raw label to processed label
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    le_dict = {key:le_dict[key] for key in le_dict}
    
    data_dict = {'docs': raw_agnews['text'], 'labels': raw_agnews['label']}
    return data_dict, le_dict
    
def load_sst2_data():
    raw_sst2 = load_dataset('glue', 'sst2', split='validation')
    label_classes = np.unique(raw_sst2['label'])
    le = preprocessing.LabelEncoder()
    le.fit(label_classes)
    # Map raw label to processed label
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    le_dict = {key:le_dict[key] for key in le_dict}
    
    data_dict = {'docs': raw_sst2['sentence'], 'labels': raw_sst2['label']}
    return data_dict, le_dict
    

def load_path_data(data_path, tokenizer):
    data = readJson(data_path)
    # Clean reports
    data = cleanSplit(data, stripChars)
    data['dev_test'] = cleanReports(data['dev_test'], stripChars)
    data = fixLabel(data)
    print("Processing train data...")
    train_documents = [extract_synoptic(patient['document'].lower(), tokenizer) for patient in data['train']]
    print("Processing val data...")
    val_documents = [extract_synoptic(patient['document'].lower(), tokenizer) for patient in data['val']]
    print("Processing test data...")
    test_documents = [extract_synoptic(patient['document'].lower(), tokenizer) for patient in data['test']]
    
    # Create datasets
    train_labels = [patient['labels']['PrimaryGleason'] for patient in data['train']]
    val_labels = [patient['labels']['PrimaryGleason'] for patient in data['val']]
    test_labels = [patient['labels']['PrimaryGleason'] for patient in data['test']]

    train_documents, train_labels = exclude_labels(train_documents, train_labels)
    val_documents, val_labels = exclude_labels(val_documents, val_labels)
    test_documents, test_labels = exclude_labels(test_documents, test_labels)

    le = preprocessing.LabelEncoder()
    le.fit(train_labels)

    # Map raw label to processed label
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    le_dict = {str(key):le_dict[key] for key in le_dict}

    for label in val_labels + test_labels:
        if str(label) not in le_dict:
            le_dict[str(label)] = len(le_dict)

    # Map processed label back to raw label
    inv_le_dict = {v: k for k, v in le_dict.items()}
    
    #docs_dict = {'train': train_documents, 'val': val_documents, 'test': test_documents}
    #labels_dict = {'train': train_labels, 'val': val_labels, 'test': test_labels}
    data_dict = {'docs': test_documents, 'labels': test_labels}
    
    return data_dict, le_dict

def load_path_model(model_checkpoint_file, le_dict):
    print("Loading in model...")
    model = BertForSequenceClassification.from_pretrained(model_checkpoint_file, num_labels=len(le_dict), output_hidden_states=True)
    return model
    