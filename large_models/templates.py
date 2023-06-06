class Template:
    def encode(self, sample):
        """
        Return prompted version of the example (without the answer/candidate)
        """
        raise NotImplementedError
    
    def verbalize(self, sample, candidate):
        """
        Return the prompted version of the example (with the answer/candidate)
        """
        return candidate
    
    def encode_sfc(self, sample):
        """
        Same as encode, but for SFC (calibration) -- this usually means the input is not included
        """
        return "<mask>"
    
    def verbalize_sfc(self, sample, candidate):
        """
        Same as verbalize, but for SFC (calibration) -- this usually means the input is not included
        """
        return candidate


class SST2Template(Template):
    verbalizer = {0: "terrible", 1: "great"}
    def encode(self, sample):
        text = sample.data["sentence"].strip()
        return f"{text} It was"

    def verbalize(self, sample, candidate):
        text = sample.data["sentence"].strip()
        return f"{text} It was {self.verbalizer[candidate]}"
    
    def encode_sfc(self, sample):
        return f" It was"

    def verbalize_sfc(self, sample, candidate):
        return f" It was {self.verbalizer[candidate]}"


class CopaTemplate(Template):
    capitalization: str = "correct"
    effect_conj: str = " so "
    cause_conj: str = " because "

    def get_conjucture(self, sample):
        if sample.data["question"] == "effect":
            conjunction = self.effect_conj
        elif sample.data["question"] == "cause":
            conjunction = self.cause_conj
        else:
            raise NotImplementedError
        return conjunction
    
    def get_prompt(self, sample):
        premise = sample.data["premise"].rstrip()
        if premise.endswith("."):  # TODO Add other scripts with different punctuation
            premise = premise[:-1]
        conjunction = self.get_conjucture(sample)
        prompt = premise + conjunction
        if self.capitalization == "upper":
            prompt = prompt.upper()
        elif self.capitalization == "lower":
            prompt = prompt.lower()
        return prompt

    def encode(self, sample):
        prompt = self.get_prompt(sample)
        return prompt 

    def capitalize(self, c):
        if self.capitalization == "correct":
            words = c.split(" ")
            if words[0] != "I":
                words[0] = words[0].lower()
            return " ".join(words)
        elif self.capitalization == "bug":
            return c
        elif self.capitalization == "upper":
            return c.upper()
        elif self.capitalization == "lower":
            return c.lower()
        else:
            raise NotImplementedError
            
    def verbalize(self, sample, candidate):
        prompt = self.get_prompt(sample)
        return prompt + self.capitalize(candidate)
    
    def encode_sfc(self, sample):
        conjunction = self.get_conjucture(sample)
        return conjunction.strip() 

    def verbalize_sfc(self, sample, candidate):
        conjunction = self.get_conjucture(sample)
        sfc_prompt = conjunction.strip() + " " + self.capitalize(candidate)
        return sfc_prompt
        
    
class BoolQTemplate(Template):
    def encode(self, sample):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}"

    def verbalize(self, sample, candidate):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question} {candidate}"
    
    def encode_sfc(self, sample):
        return ""
    
    def verbalize_sfc(self, sample, candidate):
        return candidate


class BoolQTemplateV2(Template):
    def encode(self, sample):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\\n\\n"

    def verbalize(self, sample, candidate):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\\n\\n{candidate}"
    
    def encode_sfc(self, sample):
        return ""
    
    def verbalize_sfc(self, sample, candidate):
        return candidate


class BoolQTemplateV3(Template):
    def encode(self, sample):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\n"

    def verbalize(self, sample, candidate):
        passage = sample.data["passage"]
        question = sample.data["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        return f"{passage} {question}\n{candidate}"
    
    def encode_sfc(self, sample):
        return ""
    
    def verbalize_sfc(self, sample, candidate):
        return candidate
    

class MultiRCTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        paragraph = sample.data["paragraph"]
        question = sample.data["question"]
        answer = sample.data["answer"]
        return f"{paragraph}\nQuestion: {question}\nI found this answer \"{answer}\". Is that correct? Yes or No?\n"

    def verbalize(self, sample, candidate):
        paragraph = sample.data["paragraph"]
        question = sample.data["question"]
        answer = sample.data["answer"]
        return f"{paragraph}\nQuestion: {question}\nI found this answer \"{answer}\". Is that correct? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"

    
class CBTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "Yes", 1: "No", 2: "Maybe"}

    def encode(self, sample):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Suppose {premise} Can we infer that \"{hypothesis}\"? Yes, No, or Maybe?\n"

    def verbalize(self, sample, candidate):
        premise = sample.data["premise"]
        hypothesis = sample.data["hypothesis"]
        return f"Suppose {premise} Can we infer that \"{hypothesis}\"? Yes, No, or Maybe?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class WICTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        sent1 = sample.data["sentence1"]
        sent2 = sample.data["sentence2"]
        word = sample.data["word"]
        return f"Does the word \"{word}\" have the same meaning in these two sentences? Yes, No?\n{sent1}\n{sent2}\n"

    def verbalize(self, sample, candidate):
        sent1 = sample.data["sentence1"]
        sent2 = sample.data["sentence2"]
        word = sample.data["word"]
        return f"Does the word \"{word}\" have the same meaning in these two sentences? Yes, No?\n{sent1}\n{sent2}\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class WSCTemplate(Template):
    # From PromptSource 1
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample):
        text = sample.data['text']
        span1 = sample.data['span1_text']
        span2 = sample.data['span2_text']
        return f"{text}\nIn the previous sentence, does the pronoun \"{span2.lower()}\" refer to {span1}? Yes or No?\n"

    def verbalize(self, sample, candidate):
        text = sample.data['text']
        span1 = sample.data['span1_text']
        span2 = sample.data['span2_text']
        return f"{text}\nIn the previous sentence, does the pronoun \"{span2.lower()}\" refer to {span1}? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class ReCoRDTemplate(Template):
    # From PromptSource 1 but modified

    def encode(self, sample):
        passage = sample.data['passage']
        query = sample.data['query']
        return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer:"

    def verbalize(self, sample, candidate):
        passage = sample.data['passage']
        query = sample.data['query']
        return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer: {candidate}"

    def encode_sfc(self, sample):
        return f"Answer:"

    def verbalize_sfc(self, sample, candidate):
        return f"Answer: {candidate}"


class ReCoRDTemplateGPT3(Template):
    # From PromptSource 1 but modified

    def encode(self, sample):
        passage = sample.data['passage'].replace("@highlight\n", "- ")
        return f"{passage}\n-"

    def verbalize(self, sample, candidate):
        passage = sample.data['passage'].replace("@highlight\n", "- ")
        query = sample.data['query'].replace("@placeholder", candidate[0] if isinstance(candidate, list) else candidate)
        return f"{passage}\n- {query}"

        # passage = sample.data['passage']
        # query = sample.data['query']
        # return f"{passage}\n{query}\nQuestion: what is the \"@placeholder\"\nAnswer: {candidate}"

    def encode_sfc(self, sample):
        return f"-"

    def verbalize_sfc(self, sample, candidate):
        query = sample.data['query'].replace("@placeholder", candidate[0] if isinstance(candidate, list) else candidate)
        return f"- {query}"


class RTETemplate(Template):
    # From PromptSource 1
    verbalizer={0: "Yes", 1: "No"}

    def encode(self, sample):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\nDoes this mean that \"{hypothesis}\" is true? Yes or No?\n"

    def verbalize(self, sample, candidate):
        premise = sample.data['premise']
        hypothesis = sample.data['hypothesis']
        return f"{premise}\nDoes this mean that \"{hypothesis}\" is true? Yes or No?\n{self.verbalizer[candidate]}"

    def encode_sfc(self, sample):
        return f""

    def verbalize_sfc(self, sample, candidate):
        return f"{self.verbalizer[candidate]}"


class SQuADv2Template(Template):

    def encode(self, sample):
        question = sample.data['question'].strip()
        title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0] # there are multiple answers. for the prompt we only take the first one

        return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample, candidate):
        question = sample.data['question'].strip()
        title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0] # there are multiple answers. for the prompt we only take the first one

        return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer: {answer}\n"

    
    def encode_sfc(self, sample):
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        raise NotImplementedError


class DROPTemplate(Template):

    def encode(self, sample):
        question = sample.data['question'].strip()
        # title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0] # there are multiple answers. for the prompt we only take the first one

        return f"Passage: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample, candidate):
        question = sample.data['question'].strip()
        # title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0] # there are multiple answers. for the prompt we only take the first one

        return f"Passage: {context}\nQuestion: {question}\nAnswer: {answer}\n"

    
    def encode_sfc(self, sample):
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        raise NotImplementedError
