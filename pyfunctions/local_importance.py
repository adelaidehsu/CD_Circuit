import os
import numpy as np
from methods.bag_of_ngrams.processing import cleanReports, cleanSplit, stripChars
from pyfunctions.config import BASE_DIR
from pyfunctions.general import extractListFromDic, readJson
from pyfunctions.pathology import extract_synoptic, fixLabel, exclude_labels
from pyfunctions.cdt_basics import comp_cd_scores_level_skip
from sklearn import preprocessing
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

############ MAIN FUNCTIONS TO CALL ############
def load_data(data_name, device, model_type):
    if data_name == "pathology":
        if model_type == "bert":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif model_type == "pubmed_bert":
            tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        data_path = os.path.join(BASE_DIR, "data/prostate.json")
        data, le_dict = load_path_data(data_path, tokenizer)
    elif data_name == "sst2":
        tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
        data, le_dict = load_sst2_data()
    elif data_name == "agnews":
        tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
        data, le_dict = load_agnews_data()
    return data, le_dict, tokenizer

def load_model(model_name, le_dict, device):
    if model_name == "path_bert":
        model_path = os.path.join(BASE_DIR, "models/path/bert_PrimaryGleason")
        model_checkpoint_file = os.path.join(model_path, "save_output")
        model = load_path_model(model_checkpoint_file, le_dict)
    elif model_name == "path_pubmed_bert":
        model_path = os.path.join(BASE_DIR, "models/path/pubmed_bert_PrimaryGleason")
        model_checkpoint_file = os.path.join(model_path, "save_output")
        model = load_path_model(model_checkpoint_file, le_dict)
    elif model_name == "sst2":
        model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
    elif model_name == "agnews":
        model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
    model = model.eval()
    model.to(device)
    return model

def run_local_importance(text, label, max_seq_len, model, tokenizer, le_dict, method, class_names, device):
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(text)["input_ids"])
    intervals, words = compute_word_intervals(tokens)
    if method == "CDT":
        scores, irrel_scores = comp_cd_scores_level_skip(model, encoding, label, le_dict, device, max_seq_len = 512, level = 0, skip = 1, num_at_time = 64)
    elif method == "lime":
        scores = run_lime(text, class_names, model, words)
    elif method == "shap":
        scores = run_shap(text, shap_predictor, tokenizer, intervals)
#################################################
# IG
def run_ig(text, label_idx):
    ig = IntegratedGradients(predict_forward_func)
    if not \
    type(model.get_input_embeddings()).__name__ == "InterpretableEmbeddingBase":
        interpretable_embedding1, interpretable_embedding2, interpretable_embedding3 = configure_interpretable_embeddings()

    model.to(device)

    input_data, input_data_embed = get_input_data(text)
    attributions, approximation_error = ig_attribute(ig, label_idx, input_data_embed)
    scores = attributions[0].detach().cpu().numpy().squeeze().sum(1)

    # Remove interpratable embedding layer used by ig attribution
    remove_interpretable_embeddings(interpretable_embedding1,
                                  interpretable_embedding2,
                                  interpretable_embedding3)

    tokens = tokenizer.convert_ids_to_tokens(tokenizer(text)["input_ids"])
    intervals, words = compute_word_intervals(tokens)
    word_scores = combine_token_scores(intervals, scores)

    return words, word_scores

# LIME
def lime_predictor(texts, model, tokenizer, device):
    outputs = model(**tokenizer(texts, return_tensors="pt", padding=True).to(device))
    tensor_logits = outputs[0]
    probas = F.softmax(tensor_logits).detach().cpu().numpy()
    return probas

def run_lime(text, class_names, lime_predictor, words):
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, lime_predictor, num_features=20, num_samples=2000)
    scores = exp.as_list()
    score_dict = {}

    for t in scores:
        score_dict[t[0]] = t[1]

    word_scores = np.zeros(len(words))
    for i, w in enumerate(words):
        if w in score_dict:
            word_scores[i] = score_dict[w]

    return word_scores

# SHAP
def shap_predictor(texts, model, device, max_seq_len):
    model.to(device)
    tv = torch.tensor(
        [
            tokenizer.encode(v, padding="max_length", max_length=max_seq_len, truncation=True) for v in texts
        ]
    ).to(device)

    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:, 1])  # use one vs rest logit units-- positive -> TODO: enable this for multi class

    return val

def run_shap(text, shap_predictor, tokenizer, intervals):
    explainer = shap.Explainer(shap_predictor, tokenizer)
    scores = explainer([text], fixed_context=1)
    word_scores = combine_token_scores(intervals, scores.values[0])

    return word_scores

# visualization helper
def visualize_cdt(scores, irrel_scores, tokenizer, encoding):
    toks = tokenizer.convert_ids_to_tokens([x for x in encoding['input_ids'][0] if x !=0 ])
    intervals, words = compute_word_intervals(toks)

    word_scores = combine_token_scores(intervals, scores)
    irrel_word_scores = combine_token_scores(intervals, irrel_scores) # for ablation purpose
    
    normalized = normalize_word_scores(word_scores)
    irrel_normalized = normalize_word_scores(irrel_word_scores)
    
    display_colored_html(words, normalized)
    display_colored_html(words, irrel_normalized)
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
    #print("Processing train data...")
    #train_documents = [extract_synoptic(patient['document'].lower(), tokenizer) for patient in data['train']]
    print("Processing val data...")
    val_documents = [extract_synoptic(patient['document'].lower(), tokenizer) for patient in data['val']]
    #print("Processing test data...")
    #test_documents = [extract_synoptic(patient['document'].lower(), tokenizer) for patient in data['test']]
    
    # Create datasets
    train_labels = [patient['labels']['PrimaryGleason'] for patient in data['train']]
    val_labels = [patient['labels']['PrimaryGleason'] for patient in data['val']]
    test_labels = [patient['labels']['PrimaryGleason'] for patient in data['test']]

    #train_documents, train_labels = exclude_labels(train_documents, train_labels)
    val_documents, val_labels = exclude_labels(val_documents, val_labels)
    #test_documents, test_labels = exclude_labels(test_documents, test_labels)

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
    data_dict = {'docs': val_documents, 'labels': val_labels}
    
    return data_dict, le_dict

def load_path_model(model_checkpoint_file, le_dict):
    print("Loading in model...")
    model = BertForSequenceClassification.from_pretrained(model_checkpoint_file, num_labels=len(le_dict), output_hidden_states=True)
    return model
    