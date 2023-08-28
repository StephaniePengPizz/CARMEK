import torch.cuda
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelWithLMHead
from bioasq_preprocess import pro, sample
from transformers.trainer_callback import EarlyStoppingCallback
from build_dataset import build_dataset_for_bio, build_dataset_for_pubmedqa
torch.cuda.empty_cache()
def finetune_pegasus(data_name):
    if data_name == "bioasq":
        datasets = build_dataset_for_bio()
    else:
        datasets = build_dataset_for_pubmedqa()
    model_name = "/root/autodl-tmp/pegasus-qa"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Preprocess the data and tokenize the inputs
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        contexts = [c.strip() for c in examples["context"]]
        model_inputs = tokenizer(
            questions,
            contexts,
            max_length=500,
            truncation=True,
        )
        answers = [q[0].strip() for q in examples["answer"]]
        labels = tokenizer(
            answers,
            max_length=500,
            truncation=True
        )#存在不是字符串的东西
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = datasets.map(preprocess_function, batched=True)

    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        f"/root/autodl-tmp/my_qa_model2",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=20,
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)],
    )
    total_steps = training_args.num_train_epochs * (len(dataset["train"]) // training_args.per_device_train_batch_size)
    print(total_steps)
    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model("/root/autodl-tmp/my_pegasus_model_on_{}".format(data_name))

def use_pegasus(data_name):
    #question, context = sample()
    question = "where is China?"
    context = "China is in Asia."
    #question = "Are women with major depression in pregnancy identifiable in population health data?"
    #context = [
            #"Although record linkage of routinely collected health datasets is a valuable research resource, most datasets are established for administrative purposes and not for health outcomes research. In order for meaningful results to be extrapolated to specific populations, the limitations of the data and linkage methodology need to be investigated and clarified. It is the objective of this study to investigate the differences in ascertainment which may arise between a hospital admission dataset and a dispensing claims dataset, using major depression in pregnancy as an example. The safe use of antidepressants in pregnancy is an ongoing issue for clinicians with around 10% of pregnant women suffer from depression. As the birth admission will be the first admission to hospital during their pregnancy for most women, their use of antidepressants, or their depressive condition, may not be revealed to the attending hospital clinicians. This may result in adverse outcomes for the mother and infant.",
            #"Population-based de-identified data were provided from the Western Australian Data Linkage System linking the administrative health records of women with a delivery to related records from the Midwives' Notification System, the Hospital Morbidity Data System and the national Pharmaceutical Benefits Scheme dataset. The women with depression during their pregnancy were ascertained in two ways: women with dispensing records relating to dispensed antidepressant medicines with an WHO ATC code to the 3rd level, pharmacological subgroup, 'N06A Antidepressants'; and, women with any hospital admission during pregnancy, including the birth admission, if a comorbidity was recorded relating to depression.",
            #"From 2002 to 2005, there were 96698 births in WA. At least one antidepressant was dispensed to 4485 (4.6%) pregnant women. There were 3010 (3.1%) women with a comorbidity related to depression recorded on their delivery admission, or other admission to hospital during pregnancy. There were a total of 7495 pregnancies identified by either set of records. Using data linkage, we determined that these records represented 6596 individual pregnancies. Only 899 pregnancies were found in both groups (13.6% of all cases). 80% of women dispensed an antidepressant did not have depression recorded as a comorbidity on their hospital records. A simple capture-recapture calculation suggests the prevalence of depression in this population of pregnant women to be around 16%."]
    #question = "Is cytokeratin immunoreactivity useful in the diagnosis of short-segment Barrett's oesophagus in Korea?"
    #context = [
    #    "Cytokeratin 7/20 staining has been reported to be helpful in diagnosing Barrett's oesophagus and gastric intestinal metaplasia. However, this is still a matter of some controversy.",
    #    "To determine the diagnostic usefulness of cytokeratin 7/20 immunostaining for short-segment Barrett's oesophagus in Korea.",
    #    "In patients with Barrett's oesophagus, diagnosed endoscopically, at least two biopsy specimens were taken from just below the squamocolumnar junction. If goblet cells were found histologically with alcian blue staining, cytokeratin 7/20 immunohistochemical stains were performed. Intestinal metaplasia at the cardia was diagnosed whenever biopsy specimens taken from within 2 cm below the oesophagogastric junction revealed intestinal metaplasia. Barrett's cytokeratin 7/20 pattern was defined as cytokeratin 20 positivity in only the superficial gland, combined with cytokeratin 7 positivity in both the superficial and deep glands.",
    #    "Barrett's cytokeratin 7/20 pattern was observed in 28 out of 36 cases (77.8%) with short-segment Barrett's oesophagus, 11 out of 28 cases (39.3%) with intestinal metaplasia at the cardia, and nine out of 61 cases (14.8%) with gastric intestinal metaplasia. The sensitivity and specificity of Barrett's cytokeratin 7/20 pattern were 77.8 and 77.5%, respectively."
    #]
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/my_pegasus_model_on_{}".format(data_name))
    model = AutoModelForSeq2SeqLM.from_pretrained("/root/autodl-tmp/my_pegasus_model_on_{}".format(data_name))
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt", max_length=512,
                                   truncation=True)
    beam_search_outputs = model.generate(
        inputs.input_ids,
        num_beams=5,
        num_return_sequences=5,  # You can adjust this number to control the number of generated answers
        max_length=50,
    )
    # 解码答案
    candidate_answers = []
    for output in beam_search_outputs:
        candidate_answer = tokenizer.decode(output, skip_special_tokens=True)
        candidate_answer = candidate_answer.strip()
        candidate_answers.append(candidate_answer)

    print("Question:", question)
    print("Context:", context)
    print("Generated Answers:")
    for i, answer in enumerate(candidate_answers):
        print(f"Answer {i + 1}: {answer}")
finetune_pegasus("pubmedqa")
use_pegasus("pubmedqa")
#finetune_pegasus("bioasq")
#use_pegasus("bioasq")
