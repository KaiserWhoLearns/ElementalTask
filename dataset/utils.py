import csv
import os
import copy
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


class TextFRCT:
    def __init__(self, data_path='dataset/TextFRCT.csv', eval_llm='gpt-4o-mini-2024-07-18', skip_subjective=False):
        self.skip_subjective = skip_subjective
        
        if not skip_subjective:
            load_dotenv()
            self.eval_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.eval_client = None
        
        self.eval_llm = eval_llm
        
        self.data = []
        with open(data_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.data.append(row)
        self.test_ids = sorted(set([i['category_id'] for i in self.data]))
        
        # Define which categories are subjective vs objective
        self.subjective_categories = [
            'FA3', 'FE1', 'FE2', 'FE3', 'FI1', 'FI2', 'FI3', 
            'FW1', 'FW2', 'FW3', 'XU1', 'XU2', 'XU3', 'XU4'
        ]
        self.objective_categories = [
            'CV1', 'CV2', 'CV3', 'FA1', 'FA2', 'I1', 'I2', 'MA2', 'MA3', 
            'RG1', 'RG2', 'RG3', 'RL1', 'RL3', 'RL4', 'V1', 'V2', 'V3', 'V4', 'V5'
        ]

    def replace_choice_tags(self, text, additional):
        def replacer(match):
            index = int(match.group(1))
            if 0 <= index < len(additional):
                return additional[index]
            else:
                return match.group(0)
        return re.sub(r"<CHOICE_(\d+)>", replacer, text)

    def extract_last_json_answer(self, s):
        pattern = r'\{\s*"answer"\s*:\s*(.+?)\s*\}'
        matches = list(re.finditer(pattern, s))
        if not matches:
            return s

        raw_value = matches[-1].group(1).strip()

        if (raw_value.startswith('"') and raw_value.endswith('"')) or \
           (raw_value.startswith("'") and raw_value.endswith("'")):
            raw_value = raw_value[1:-1]
        return raw_value

    def merge_strings_without_colon(self, lst):
        result = []
        for s in lst:
            if ':' in s:
                result.append(s)
            else:
                if result:
                    result[-1] += s
                else:
                    result.append(s)
        return result
    
    def match_FE1(self, pattern, sentence):
        pattern_parts = pattern.strip("_").split("_")
        words = sentence.strip().split()
        if len(pattern_parts) != len(words):
            return False
        for pat, word in zip(pattern_parts, words):
            if pat != "*":
                if not word or word[0].lower() != pat.lower():
                    return False
        return True
    
    def match_FE2(self, words, sentence):
        sentence = sentence.lower().replace(',', ' ').replace('.', ' ').replace('\'', ' ')
        sentence_words = set(sentence.split())
        return all(word.lower() in sentence_words for word in words.split())
    
    def match_FW1(self, pattern, word):
        return word.lower().endswith(pattern.lower())
    
    def match_FW2(self, pattern, word):
        return word.lower().startswith(pattern.lower())
    
    def match_FW3(self, pattern, word):
        s, e = pattern.split(';;')
        return word.lower().startswith(s.lower()) and word.lower().endswith(e.lower())

    def build_prompt(self, questions, demonstrations):
        for row in self.data:
            if row['category_id'] == 'FW3':
                s, e = row['question'].split(';;')
                question = questions[row['category_id']].replace('<QUESTION_0>', s).replace('<QUESTION_1>', e)
            else:
                question = questions[row['category_id']].replace('<ADDITIONAL>', row['additional'].replace('<br>', '\n')).replace('<QUESTION>', row['question'].replace('<br>', '\n'))
            
            if row['choice'] != '':
                question = self.replace_choice_tags(question, row['choice'].split(';;'))
            
            demos = demonstrations[row['category_id']].replace('<br>', '\n')
            demos = demos.split(';;')
            for idx, d in enumerate(demos):
                # demos[idx] = f'Example {idx + 1}:\n{demos[idx]}'
                demos[idx] = f'\n{demos[idx]}\n'
            demo = ''.join(demos)
            
            prompt = question.replace('<DEMO>',demo)
            
            yield prompt
    
    def evaluate(self, raw_predictions, save_file='results.csv'):
        save_data = copy.deepcopy(self.data)
        assert len(raw_predictions) == len(save_data)
        predictions = [self.extract_last_json_answer(i) for i in raw_predictions]
        accuracy = {key: [] for key in self.test_ids}

        for idx in tqdm(range(len(save_data))):
            category_id = save_data[idx]['category_id']
            
            # Skip subjective categories if requested
            if self.skip_subjective and category_id in self.subjective_categories:
                save_data[idx]['predictions'] = raw_predictions[idx]
                save_data[idx]['processed_preds'] = ["SKIPPED_SUBJECTIVE"]
                save_data[idx]['pred_num'] = 0
                save_data[idx]['pred_correct'] = 0
                continue
            
            single_count = []
            answers = save_data[idx]['answer'].split(';;')
            preds = [i.strip() for i in predictions[idx].lower().split('\n') if i.strip()]
            preds = list(dict.fromkeys(preds))
            
            if category_id in self.objective_categories:
                for pred in preds:
                    single_count.append(pred in [i.lower() for i in answers])
            
            elif category_id in self.subjective_categories:
                # This shouldn't be reached if skip_subjective=True, but handle it
                if self.eval_client is None:
                    print(f"Warning: Skipping subjective evaluation for {category_id} (no OpenAI client)")
                    single_count = [False] * len(preds)  # Mark as incorrect
                else:
                    for pred in preds:
                        query = answers[0].replace('<LLMEval>', f'You need to decide whether "{pred}" is an acceptable answer. ')
                        query += ' Respond with only one letter: Y if the answer is acceptable, N if it is not, in JSON format as follows: {"answer": YOUR_ANSWER_HERE}.'
                        response = self.eval_client.responses.create(
                            model=self.eval_llm,
                            input=query
                        )
                        decision = self.extract_last_json_answer(response.output_text.lower())
                        match = True
                        if category_id == 'FE1':
                            match = self.match_FE1(save_data[idx]['question'], pred)
                        elif category_id == 'FE2':
                            match = self.match_FE2(save_data[idx]['question'], pred)
                        elif category_id == 'FW1':
                            match = self.match_FW1(save_data[idx]['question'], pred)
                        elif category_id == 'FW2':
                            match = self.match_FW2(save_data[idx]['question'], pred)
                        elif category_id == 'FW3':
                            match = self.match_FW3(save_data[idx]['question'], pred)
                        
                        single_count.append(decision == 'y' and match)
            
            elif category_id == 'XU3':
                # Handle XU3 separately since it has special logic
                if self.skip_subjective:
                    save_data[idx]['predictions'] = raw_predictions[idx]
                    save_data[idx]['processed_preds'] = ["SKIPPED_SUBJECTIVE"]
                    save_data[idx]['pred_num'] = 0
                    save_data[idx]['pred_correct'] = 0
                    continue
                
                parse = self.merge_strings_without_colon(preds)
                groups = [p.split(':')[0].split(',') for p in parse]
                groups = [[j.strip() for j in i] for i in groups]
                reason = [p.split(':')[1].strip() for p in parse]
                
                lenth_check = [len(i) >= 3 for i in groups]
                
                repeat_check = []
                seen = []
                for i in [','.join(i) for i in groups]:
                    if i in seen:
                        repeat_check.append(False)
                    else:
                        repeat_check.append(True)
                        seen.append(i)

                llm_check = []
                for g, r in zip(groups, reason):
                    groups_str = ', '.join(g)
                    query = answers[0].replace('<LLMEval>', f'You need to decide whether "Groups: {groups_str}; Reason: {r}" is an acceptable answer. ').replace('<br>', '\n')
                    query += 'Respond with only one letter: Y if the answer is acceptable, N if it is not, in JSON format as follows: {"answer": YOUR_ANSWER_HERE}.'
                    response = self.eval_client.responses.create(
                        model=self.eval_llm,
                        input=query
                    )
                    decision = self.extract_last_json_answer(response.output_text.lower())
                    llm_check.append(decision == 'y')
                single_count = [(i and j and k) for i, j, k in zip(lenth_check, repeat_check, llm_check)]
            
            save_data[idx]['predictions'] = raw_predictions[idx]
            save_data[idx]['processed_preds'] = preds
            save_data[idx]['pred_num'] = len(single_count)
            save_data[idx]['pred_correct'] = sum(single_count)
            
        with open(save_file, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=save_data[0].keys())
            writer.writeheader()
            writer.writerows(save_data)
        
        for row in save_data:
            cid = str(row['category_id'])
            # Skip subjective categories if they were skipped
            if self.skip_subjective and cid in self.subjective_categories:
                continue
            acc = row['pred_correct'] / row['pred_num'] if row['pred_num'] != 0 else 0
            accuracy[cid].append(acc)
        
        # Filter out empty categories (skipped subjective ones)
        filtered_accuracy = {}
        for subtest in accuracy:
            if accuracy[subtest]:  # Only include categories that have data
                filtered_accuracy[subtest] = sum(accuracy[subtest]) / len(accuracy[subtest])

        if filtered_accuracy:
            filtered_accuracy['ALL'] = sum([filtered_accuracy[s] for s in filtered_accuracy]) / len([filtered_accuracy[s] for s in filtered_accuracy])
        
        with open(save_file.replace('.csv', '_acc.csv'), 'w') as f:
            for key in filtered_accuracy:
                f.write(f'{key},{filtered_accuracy[key]}\n')
        
        for i in filtered_accuracy.keys():
            print(f"{i:3}: {filtered_accuracy[i] * 100:8.2f}")

        return filtered_accuracy