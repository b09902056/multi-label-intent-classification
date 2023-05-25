# parse arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--b", help = "batch size", type=int, default=8)
parser.add_argument("--e", help = "epoch", type=int, default=5)
parser.add_argument("--lr", help = "learning rate", type=float, default=3e-5)
parser.add_argument("--regime", help = "regime", type=str, default='low')
parser.add_argument("--fold", help = "fold", type=int, default=None)
args = parser.parse_args()

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import random
import json

from nlupp.data_loader import DataLoader
loader = DataLoader("./nlupp/data/")
banking_data = loader.get_data_for_experiment(domain="banking", regime=args.regime)

import sys
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import copy
import os
from torch.utils.data import DataLoader
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast, BertForMultipleChoice, BertConfig
from transformers.optimization import get_linear_schedule_with_warmup
from accelerate import Accelerator
import evaluate
import math
import collections
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import logging
from typing import Optional, Tuple




hidden_size = 768 # roberta-base = 768, roberta-large=1024
dropout = 0.5
batch_size = args.b
epoch = args.e
lr = args.lr
threshold = 0.5
regime = args.regime
pretrained_model = "deepset/roberta-base-squad2"
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
print(f'dropout={dropout}, batch_size={batch_size}, epoch={epoch}, LR={lr}, regime={regime}')

f = open('./nlupp/data/ontology.json')
ontology = json.load(f)
label2id = {}
label2description = {}
intents = ontology['intents']
for intent in intents:
    if 'general' in intents[intent]['domain'] or 'banking' in intents[intent]['domain']:
        label2id[intent] = len(label2id)
        label2description[intent] = intents[intent]['description']

num_intents = len(label2id) # 48

fold = len(banking_data)
if args.fold != None:
    fold = args.fold
print('folds =', fold)



example1 = {'text': 'How long does it usually take to get a new pin?', 'intents': ['how_long', 'pin', 'arrival', 'new']}
example2 = {'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
 'id': '5733be284776f41900661182',
 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
 'title': 'University_of_Notre_Dame'
}

for e in range(fold):
    train_data = []
    for j in range(len(banking_data[e]['train'])):
        for intent in label2id:
            x = banking_data[e]['train'][j]
            new_data = {}
            new_data['id'] = str(j) + intent
            new_data['question'] = label2description[intent]
            new_data['title'] = ''
            new_data['context'] = 'yes. no. ' + x['text']
            if 'intents' not in x:
                new_data['answers'] = {'answer_start': [5], 'text': ['no']}
            else:
                if intent in x['intents']:
                    new_data['answers'] = {'answer_start': [0], 'text': ['yes']}
                else:
                    new_data['answers'] = {'answer_start': [5], 'text': ['no']}
            train_data.append(new_data)
    test_data = []
    for j in range(len(banking_data[e]['test'])):
        for intent in label2id:
            x = banking_data[e]['test'][j]
            new_data = {}
            new_data['id'] = str(j) + intent
            new_data['question'] = label2description[intent]
            new_data['title'] = ''
            new_data['context'] = 'yes. no. ' + x['text']
            if 'intents' not in x:
                new_data['answers'] = {'answer_start': [5], 'text': ['no']}
            else:
                if intent in x['intents']:
                    new_data['answers'] = {'answer_start': [0], 'text': ['yes']}
                else:
                    new_data['answers'] = {'answer_start': [5], 'text': ['no']}
            test_data.append(new_data)
    valid_data = test_data


    print(f'train len = {len(train_data)} = {num_intents} * {len(banking_data[e]["train"])}')
    print(f'test len = {len(test_data)} = {num_intents} * {len(banking_data[e]["test"])}')
    # -*- coding: utf-8 -*-
    """adlhw2-QA.ipynb

    Automatically generated by Colaboratory.

    Original file is located at
        https://colab.research.google.com/drive/1lXX_7qq-G8YdViWmtlBEXwFgbbFHEIIL
    """


    validation = True
    sample = False
    sampleN = 100
    gradient_accum = 4
    # pretrain_model = "hfl/chinese-macbert-large"
    # pretrain_model = "hfl/chinese-macbert-base"
    # pretrain_model = "bert-base-chinese"
    pretrain_model = "deepset/roberta-base-squad2"
    max_len = 512
    print('gradient_accum =', gradient_accum)
    print('pretrain_model =', pretrain_model)
    print('max_len =', max_len)
    if sample:
        train_data = train_data[:sampleN]
        test_data = test_data[:sampleN]
    print('train len =', len(train_data))
    print('test len =', len(test_data))




    train_dataset = Dataset.from_list(train_data)
    valid_dataset = Dataset.from_list(valid_data)
    test_dataset = Dataset.from_list(test_data)
    dataset = DatasetDict({'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset})

    accelerator_log_kwargs = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accum, **accelerator_log_kwargs)
    device = accelerator.device
    print(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(pretrain_model).to(device)
    # config = BertConfig()
    # model = AutoModelForQuestionAnswering.from_config(config).to(device)

    column_names = dataset["train"].column_names
    print('-------------', column_names)
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"
    def prepare_train_features(examples):
            # Some of the questions have lots of whitespace on the left, which is not useful and will make the
            # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
            # left whitespace
            examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            tokenized_examples = tokenizer(
                examples[question_column_name ],
                examples[context_column_name],
                truncation="only_second",
                max_length=max_len, #defalut = 384
                stride=128, #defalut = 128
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length"
            )

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offset_mapping = tokenized_examples.pop("offset_mapping")

            # Let's label those examples!
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                # We will label impossible answers with the index of the CLS token.
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(tokenizer.cls_token_id)

                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                answers = examples[answer_column_name][sample_index]
                # If no answers are given, set the cls_index as answer.
                if len(answers["answer_start"]) == 0:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Start/end character index of the answer in the text.
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    # Start token index of the current span in the text.
                    token_start_index = 0
                    while sequence_ids[token_start_index] != 1:
                        token_start_index += 1

                    # End token index of the current span in the text.
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != 1:
                        token_end_index -= 1

                    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                        tokenized_examples["start_positions"].append(cls_index)
                        tokenized_examples["end_positions"].append(cls_index)
                    else:
                        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                        # Note: we could go after the last offset if the answer is the last word (edge case).
                        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                            token_start_index += 1
                        tokenized_examples["start_positions"].append(token_start_index - 1)
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_examples["end_positions"].append(token_end_index + 1)

            return tokenized_examples


    """def prepare_validation_features(examples):"""
    def prepare_validation_features(examples):
            # Some of the questions have lots of whitespace on the left, which is not useful and will make the
            # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
            # left whitespace
            examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            tokenized_examples = tokenizer(
                examples[question_column_name],
                examples[context_column_name],
                truncation="only_second",
                max_length=max_len,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id and we will store the offset mappings.
            tokenized_examples["example_id"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])

                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]

            return tokenized_examples

    train_examples = dataset["train"]
    train_dataset2 = train_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        load_from_cache_file=not 'store_true',
        desc="Running tokenizer on train dataset 2",
    )

    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        load_from_cache_file=not 'store_true',
        desc="Running tokenizer on train dataset",
    )
    # eval_examples = dataset["valid"]
    # eval_dataset = eval_examples.map(
    #     prepare_validation_features,
    #     batched=True,
    #     num_proc=1,
    #     remove_columns=column_names,
    #     load_from_cache_file=not 'store_true',
    #     desc="Running tokenizer on validation dataset",
    # )

    predict_examples = dataset['test']
    predict_dataset = predict_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        load_from_cache_file=not 'store_true',
        desc="Running tokenizer on prediction dataset",
    )


    data_collator = default_data_collator
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size)
    # eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    # eval_dataloader = DataLoader(eval_dataset_for_model, collate_fn=data_collator, batch_size=batch_size)
    predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
    predict_dataloader = DataLoader(predict_dataset_for_model, collate_fn=data_collator, batch_size=batch_size)

    """postprocess_qa_predictions"""

    #@title
    def postprocess_qa_predictions(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = './output_dir',
        prefix: Optional[str] = None,
        log_level: Optional[int] = logging.WARNING,
    ):
        if len(predictions) != 2:
            raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
        all_start_logits, all_end_logits = predictions

        if len(predictions[0]) != len(features):
            raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        if version_2_with_negative:
            scores_diff_json = collections.OrderedDict()

        # Logging.
        #logger.setLevel(log_level)
        #logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_prediction = None
            prelim_predictions = []

            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]
                # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
                # available in the current feature.
                token_is_max_context = features[feature_index].get("token_is_max_context", None)

                # Update minimum null prediction.
                feature_null_score = start_logits[0] + end_logits[0]
                if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                    min_null_prediction = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                    }

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or len(offset_mapping[start_index]) < 2
                            or offset_mapping[end_index] is None
                            or len(offset_mapping[end_index]) < 2
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue
                        # Don't consider answer that don't have the maximum context available (if such information is
                        # provided).
                        if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                            continue

                        prelim_predictions.append(
                            {
                                "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )
            if version_2_with_negative and min_null_prediction is not None:
                # Add the minimum null prediction
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            # Only keep the best `n_best_size` predictions.
            predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

            # Add back the minimum null prediction if it was removed because of its low score.
            if (
                version_2_with_negative
                and min_null_prediction is not None
                and not any(p["offsets"] == (0, 0) for p in predictions)
            ):
                predictions.append(min_null_prediction)

            # Use the offsets to gather the answer text in the original context.
            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0] : offsets[1]]

            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

            # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
            # the LogSumExp trick).
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our predictions.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # Pick the best prediction. If the null answer is not possible, this is easy.
            if not version_2_with_negative:
                all_predictions[example["id"]] = predictions[0]["text"]
            else:
                # Otherwise we first need to find the best non-empty prediction.
                i = 0
                while predictions[i]["text"] == "":
                    i += 1
                best_non_null_pred = predictions[i]

                # Then we compare to the null prediction using the threshold.
                score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
                scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
                if score_diff > null_score_diff_threshold:
                    all_predictions[example["id"]] = ""
                else:
                    all_predictions[example["id"]] = best_non_null_pred["text"]

            # Make `predictions` JSON-serializable by casting np.float back to float.
            all_nbest_json[example["id"]] = [
                {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
                for pred in predictions
            ]

        return all_predictions

    """post_processing_function"""

    #@title
    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative="store_true",
            n_best_size=20,
            max_answer_length=30,
            null_score_diff_threshold=0.0,
            output_dir='.',
            prefix=stage,
        )
        # Format the result to the format the metric expects.

        #formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()]
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
            step = 0
            # create a numpy array and fill it with -100.
            logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
            # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
            for i, output_logit in enumerate(start_or_end_logits):  # populate columns
                # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
                # And after every iteration we have to change the step

                batch_size = output_logit.shape[0]
                cols = output_logit.shape[1]

                if step + batch_size < len(dataset):
                    logits_concat[step : step + batch_size, :cols] = output_logit
                else:
                    logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

                step += batch_size

            return logits_concat

    metric = evaluate.load("squad")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_train_epochs = epoch
    gradient_accumulation_steps = gradient_accum
    #num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    #max_train_steps = num_train_epochs * num_update_steps_per_epoch
    total_steps = num_train_epochs * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)
    print(total_steps)
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    #print(f"  Total optimization steps = {max_train_steps}")

    #progress_bar = tqdm(range(max_train_steps))
    completed_steps = 0
    starting_epoch = 0
    logging_step = 100

    loss_points, em_points = [], []
    all_start_logits, all_end_logits = [], []
    loss2 = 0

    model = model.to(device)
    for epoch in range(num_train_epochs):
        model.train()
        train_loss = train_acc = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
                with accelerator.accumulate(model):

                    batch = {b: batch[b].to(device) for b in batch}
                    start_positions = batch['start_positions']
                    end_positions = batch['end_positions']
                    outputs = model(**batch)
                    loss = outputs.loss


                    start_index = torch.argmax(outputs.start_logits, dim=1)
                    end_index = torch.argmax(outputs.end_logits, dim=1)

                    train_acc += ((start_index == start_positions) & (end_index == end_positions)).float().mean()
                    train_loss += outputs.loss

                    accelerator.backward(outputs.loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                            # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    completed_steps += 1

                if step % logging_step == 0:
                    print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                    train_loss = train_acc = 0


    model_save_dir = "./QA-model"
    torch.save(model.state_dict(), model_save_dir)
    print(f'\nsave model to {model_save_dir}')


    # print("***** Running Evaluation *****")
    # print(f"  Num examples = {len(eval_dataset)}")
    # print(f"  Batch size = {batch_size}")

    # all_start_logits = []
    # all_end_logits = []
    # # progress_bar = tqdm(range(len(eval_dataloader)))
    # model.eval()
    # for step, batch in enumerate(tqdm(eval_dataloader)):
    #     with torch.no_grad():
    #         batch = {b: batch[b].to(device) for b in batch}
    #         outputs = model(**batch)
    #         start_logits = outputs.start_logits
    #         end_logits = outputs.end_logits

    #         # progress_bar.update(1)

    #         all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
    #         all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

    # max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor # 384
    # start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
    # end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

    # outputs_numpy = (start_logits_concat, end_logits_concat)
    # prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
    # eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
    # print(f"Evaluation metrics: {eval_metric}")

    print("***** Running Prediction *****")
    print(f"  Num examples = {len(predict_dataset)}")
    print(f"  Batch size = {batch_size}")

    all_start_logits = []
    all_end_logits = []

    model.eval()
    # progress_bar = tqdm(range(len(predict_dataloader)))
    for step, batch in enumerate(tqdm(predict_dataloader)):
        with torch.no_grad():
            batch = {b: batch[b].to(device) for b in batch}
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy)
    predict_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
    print(f"Predict metrics: {predict_metric}")

    submit = {"id": [], "answer": []}

    for i in range(len(prediction.predictions)):
        submit['id'].append(test_data[i]['id'])
        submit['answer'].append(prediction.predictions[i]['prediction_text'])


    df = pd.DataFrame(submit)
    path = "./low_QA_prediction.csv"
    df.to_csv(path, index=False)
    print(f'save to {path}')
    print('gradient_accum =', gradient_accum)
    print('pretrain_model =', pretrain_model)
    print('max_len =', max_len)
    print('epoch =', epoch)
    print('lr =', lr)
    print('batch_size =', batch_size)