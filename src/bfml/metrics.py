import logging
from sklearn.metrics import f1_score, accuracy_score
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import accuracy_score as seqeval_accuracy
def xnli_metrics(eval_pred):

    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    accuracy = accuracy_score(y_true=labels, y_pred=preds)


    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
    }

def create_id2label_ner():

    ner_tags = [
        'B-ORG',
        'I-ORG',
        'B-PER',
        'I-PER',
        'B-MISC',
        'I-MISC',
        'B-LOC',
        'I-LOC',
        'O'
    ]

    iter = 0
    id2label = {}
    for tag in ner_tags:
        id2label[iter] = tag
        iter += 1

    return id2label

def ner_metrics(eval_pred):

    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    corrected_preds = []
    corrected_labels = []

    id2label = create_id2label_ner()

    for i in range(0, len(labels)):
        temp_pred = []
        temp_label = []
        for j in range(0, len(labels[i])):
            if labels[i][j] != -100:
                temp_label.append(id2label[labels[i][j]])
                temp_pred.append(id2label[preds[i][j]])

        corrected_labels.append(temp_label)
        corrected_preds.append(temp_pred)

    acc = seqeval_accuracy(corrected_labels, corrected_preds)
    f1 = seqeval_f1(corrected_labels, corrected_preds)

    f1 = f1 * 100
    acc = acc * 100

    logging.info('F1 during training: {}'.format(f1))
    logging.info('Accuracy during training: {}'.format(acc))
    logging.info('---------------------------------------------')

    return {
        'accuracy': acc,
        'f1': f1
    }

