# “I’d rather just go to bed”: Understanding Indirect Answers

---

## Open In Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1shG5ti8V6KESLgPVr6HrW_JAXpaYnr24)

## Set up the Environment

1. `sudo apt-get update && sudo apt-get upgrade -y`
2. `python -m venv venv`  
3. `source venv/bin/activate` 
4. `python -m pip install --upgrade pip` 
5. `pip install -r requirements.txt`

---

[“I’d rather just go to bed”: Understanding Indirect Answers](https://aclanthology.org/2020.emnlp-main.601) (Louis et al., EMNLP 2020)

    @inproceedings{louis-etal-2020-id,
        title = "{``}{I}{'}d rather just go to bed{''}: Understanding Indirect Answers",
        author = "Louis, Annie  and
          Roth, Dan  and
          Radlinski, Filip",
        booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
        month = nov,
        year = "2020",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2020.emnlp-main.601",
        doi = "10.18653/v1/2020.emnlp-main.601",
        pages = "7411--7425",
        abstract = "We revisit a pragmatic inference problem in dialog: Understanding indirect responses to questions. Humans can interpret {`}I{'}m starving.{'} in response to {`}Hungry?{'}, even without direct cue words such as {`}yes{'} and {`}no{'}. In dialog systems, allowing natural responses rather than closed vocabularies would be similarly beneficial. However, today{'}s systems are only as sensitive to these pragmatic moves as their language model allows. We create and release the first large-scale English language corpus {`}Circa{'} with 34,268 (polar question, indirect answer) pairs to enable progress on this task. The data was collected via elaborate crowdsourcing, and contains utterances with yes/no meaning, as well as uncertain, middle-ground, and conditional responses. We also present BERT-based neural models to predict such categories for a question-answer pair. We find that while transfer learning from entailment works reasonably, performance is not yet sufficient for robust dialog. Our models reach 82-88{\%} accuracy for a 4-class distinction, and 74-85{\%} for 6 classes.",
    }
