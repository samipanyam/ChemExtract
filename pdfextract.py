import os
import fitz
import cv2
import numpy as np
import torch
import re
import asyncio
import pubchempy as pcp
from pdf2image import convert_from_path
from molscribe import MolScribe
from decimer_segmentation import segment_chemical_structures, get_mrcnn_results
from transformers import AutoTokenizer, BertForTokenClassification
from collections import defaultdict
from  huggingface_hub import hf_hub_download

import json


async def fetch_from_pcp(keyword, type, output):
    try:
        compound = pcp.get_compounds(keyword, type)[0]
        return compound.canonical_smiles
    except pcp.PubChemHTTPError as e:
        if hasattr(e, 'args') and len(e.args) > 0:
            error_name = e.args[0]
            if error_name == 'PUGREST.ServerBusy':
                print(f"Rate limit exceeded. Retrying after {e.headers['Retry-After']} seconds...")
                retry_after = int(e.headers['Retry-After'])
                await asyncio.sleep(retry_after)
                return await fetch_from_pcp(keyword, type, output)
    except (IndexError, KeyError):
        return None
    except Exception as e:
        print(f"Error fetching compound smiles: {e}")
        return None


def extract_text_from_pdf(file_path: str, page_number: int = 0) -> str:
    doc = fitz.open(file_path)
    page = doc.load_page(page_number)
    text = page.get_text("text")
    return text


def extract_text_from_pdf_all_pages(file_path: str) -> list:
    doc = fitz.open(file_path)
    text_pages = []

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text("text")
        text_pages.append(text)

    return text_pages


def split_text(text: str, max_length: int) -> list:
    # Split the text into chunks of maximum length
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    return chunks


class StructureExtractor:
    def __init__(self, filename: str):
        self.filename = filename
        self.filename_without_extension = os.path.splitext(self.filename)[0]
        ckpt_path =  hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth')
        self.model = MolScribe(ckpt_path, device=torch.device('cpu'))
        self.pngs = []
        self.segments = []
        self.smiles = []

    def PDFtoPNG(self):
        if not self.pngs:
            pngs = convert_from_path(self.filename)
            folder_path = 'Page_PNGS'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            for index, page in enumerate(pngs):
                extracted_page = np.array(page)
                png_path = f'{folder_path}/{os.path.basename(self.filename_without_extension)}_{index}.png'
                cv2.imwrite(png_path, extracted_page)
                self.pngs.append(extracted_page)

        return self.pngs

    def segment(self):
        if not self.segments:
            folder_path = 'segments'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            if not self.pngs:
                self.PDFtoPNG()

            for index, img in enumerate(self.pngs):
                minisegments = segment_chemical_structures(img, expand=False)
                _, bounding_boxes, _ = get_mrcnn_results(img)
                averages = [(row[0] + row[2]) / 2 for row in bounding_boxes] + [(row[1] + row[3]) / 2 for row in bounding_boxes]
                grouped_averages = [[averages[i], averages[i + len(bounding_boxes)]] for i in range(len(bounding_boxes))]

                minis = []
                for idx, minisegment in enumerate(minisegments):
                    minisegment = [minisegment, idx, grouped_averages[idx], os.path.basename(self.filename_without_extension)]
                    minis.append(minisegment)
                    cv2.imwrite(f'{folder_path}/{os.path.basename(self.filename_without_extension)}_{idx}.png', minisegment[0])

                self.segments.extend(minis)

        return self.segments

    def toSMILES(self):
        if not self.smiles:
            output = []
            folder_path = 'SMILES'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            subfolder = 'PDF_SMILES'

            if not os.path.exists(f'{folder_path}/{subfolder}'):
                os.makedirs(f'{folder_path}/{subfolder}')
            
            smiles_txt_path = f'{folder_path}/{subfolder}/{os.path.basename(self.filename_without_extension)}.json'
            
            if os.path.exists(smiles_txt_path):
               
                return json.load(open(smiles_txt_path, 'r'))
            
            if not self.segments:
                self.segment()

            

            for i, img in enumerate(self.segments):
                SMILES = self.model.predict_image(img[0], return_atoms_bonds=True, return_confidence=True)['smiles']
              
                try:
                    cid = pcp.get_compounds(SMILES, 'smiles')[0].cid
                except pcp.BadRequestError:
                    cid = None
                
                
                
                to_add = {
                    'SMILES': SMILES,
                    'page': img[1],
                    'cid': cid,
                    'X': img[2][0],
                    'Y': img[2][1],
                    'article': img[3],
                    'image': f'segments/{os.path.basename(self.filename_without_extension)}_{i}.png'
                }
                output.append(to_add)

            self.smiles = output
            with open(smiles_txt_path, 'w') as f:
                json.dump(self.smiles, f)
            

        return self.smiles


class TextExtractor:
    tokenizer = AutoTokenizer.from_pretrained('pruas/BENT-PubMedBERT-NER-Chemical')
    model = BertForTokenClassification.from_pretrained('pruas/BENT-PubMedBERT-NER-Chemical')

    def __init__(self, filename: str):
        self.filename = filename
        self.filename_without_extension = os.path.splitext(self.filename)[0]
        self.keywords = None
        self.text = None
        self.total_text = None
        self.preprocessed_keywords = []
    def find_all_occurrences(self,input_string, substring):
        occurrences = []
        start_index = 0

        while True:
            index = input_string.find(substring, start_index)
            if index == -1:
                break
            occurrences.append(index)
            start_index = index + 1
        return occurrences


    def extract(self) -> list:
        if self.text is None:
            self.text = extract_text_from_pdf_all_pages(self.filename)
            delimiter = ' '
            self.total_text = delimiter.join(self.text)

        return self.text

    async def process_page(self, page_text):
        # Asynchronously process a single page's keywords
        all_keywords_with_page_number_and_index = defaultdict(list)
        split_page = split_text(page_text, 800)
        for text_splice in split_page:
            inputs = self.tokenizer.encode_plus(text_splice, return_tensors="pt", add_special_tokens=True)
            outputs = self.model(inputs.input_ids, attention_mask=inputs.attention_mask)
            predicted_labels = torch.argmax(outputs.logits, dim=2)[0]
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            keywords = []
            for token, label_idx in zip(predicted_tokens, predicted_labels):
                    if label_idx != 0:
                        keywords.append(token)
            self.preprocessed_keywords.append(keywords) 
            # combined_tokens = []
            # for token in keywords:
            #     if token.startswith("##") and combined_tokens:
            #         combined_tokens[-1] += token[2:]
            #     else:
            #         combined_tokens.append(token)
            
            words =text_splice.split()
            
            new_keywords = []
            for token in keywords:
                if token.startswith("##"):
                    partial_word = token[2:]
                    
                    if len(partial_word) > 2 and partial_word.isnumeric() == False:
                       
                        matches = [word for word in words if partial_word in word]   
                        for match in matches:
                            new_keywords.append(match)
                else:
                    new_keywords.append(token)

                   
            keywords = new_keywords
            
           
            for token in keywords:
                
                    keyword = token.strip()
                  
                    if keyword in self.total_text and len(keyword)>3:
                        
                       
                        
                        start_index = self.find_all_occurrences(self.total_text, keyword)
                        end_index = [index + len(keyword) for index in start_index]
                        indeces = list(zip(start_index, end_index))


                        all_keywords_with_page_number_and_index[keyword].append((
                            keyword,
                            self.text.index(page_text),
                            indeces
                            
                        ))

        flattened_keywords = [item for sublist in all_keywords_with_page_number_and_index.values() for item in sublist]
        
        unique_keywords = []
        seen_keywords = set()

        for entry in flattened_keywords:
            
            if entry[0] not in seen_keywords:
                unique_keywords.append({
                    "keyword": entry[0],
                    "page": entry[1],
                    "index": entry[2]
                })
                seen_keywords.add(entry[0])

        return unique_keywords



    async def getKeywords(self):
        if self.keywords is None:
            text = self.extract()
            delimiter = ' '
            self.total_text = delimiter.join(text)

            # Asynchronously process keywords for each page
            page_tasks = [self.process_page(page) for page in text]
            all_keywords = await asyncio.gather(*page_tasks)

            # Combine keywords from all pages
            self.keywords = [keyword for page_keywords in all_keywords for keyword in page_keywords]
       
        return self.keywords

    async def toSMILES(self) -> list:
        folder_path = 'SMILES'
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        subfolder = 'TEXT_SMILES'
        if not os.path.exists(f'{folder_path}/{subfolder}'):
            os.makedirs(f'{folder_path}/{subfolder}')
        if os.path.exists(f'{folder_path}/{subfolder}/{os.path.basename(self.filename_without_extension)}.json'):
           
           return json.loads(open(f'{folder_path}/{subfolder}/{os.path.basename(self.filename_without_extension)}.json', 'r').read())
        if self.keywords is None:
           await self.getKeywords()
        

        
            

        async def fetch_compound_smiles(keyword):
            try:
                compound = pcp.get_compounds(keyword["keyword"], 'name')[0]
                return compound.canonical_smiles
            except pcp.PubChemHTTPError as e:
                if hasattr(e, 'args') and len(e.args) > 0:
                    error_name = e.args[0]
                    if error_name == 'PUGREST.ServerBusy':
                        print(f"Rate limit exceeded. Retrying after {e.headers['Retry-After']} seconds...")
                        retry_after = int(e.headers['Retry-After'])
                        await asyncio.sleep(retry_after)
                        return await fetch_compound_smiles(keyword)
            except (IndexError, KeyError):
                return None
            except Exception as e:
                print(f"Error fetching compound smiles: {e}")
                return None

        tasks = [fetch_compound_smiles(keyword) for keyword in self.keywords]
        filtered_SMILES= await asyncio.gather(*tasks)

        

        tor = []
        for SMILES in filtered_SMILES:
            try:
                cid = pcp.get_compounds(SMILES, 'smiles')[0].cid
            except ValueError:
                cid = None
            except pcp.BadRequestError:
                cid = None
          
            tor.append({
                "keyword": self.keywords[filtered_SMILES.index(SMILES)]["keyword"],
                "SMILES": SMILES,
                "cid": cid,
                "page": self.keywords[filtered_SMILES.index(SMILES)]["page"],
                "index": self.keywords[filtered_SMILES.index(SMILES)]["index"]
            })

        filtered_keywords = [keyword for keyword in tor if keyword["SMILES"] is not None]

        with open(f'{folder_path}/{subfolder}/{os.path.basename(self.filename_without_extension)}.json', 'w') as f:
            json.dump(filtered_keywords, f)
        return filtered_keywords


class BatchExtractor:
    SMILES = None
    def __init__(self, path: str):
        if os.path.isdir(path):
            self.pdf_list = []
            self.text_list =[]

            files = os.listdir(path)
            for filename in files:
                if filename.endswith(".pdf"):
                    print(f'{path}/{filename}')
                    self.pdf_list.append(StructureExtractor(f'{path}/{filename}'))
                    self.text_list.append(TextExtractor(f'{path}/{filename}'))
            
         

        elif os.path.isfile(path):
            if path.endswith(".pdf"):
                self.pdf_list = [StructureExtractor(path)]
                self.text_list = [TextExtractor(path)]
           


    async def toSMILES(self):
        tor = {}
        tor["PDF_SMILES"] = []
        tor["Text_SMILES"] = []

        for extractor in self.pdf_list:
            pdf_smiles = extractor.toSMILES()
            tor["PDF_SMILES"].append(pdf_smiles)  # Await the asynchronous call

        for extractor in self.text_list:
            text_smiles = await extractor.toSMILES()
            tor["Text_SMILES"].append(text_smiles)

        self.SMILES = tor
        return tor



    async def combine(self):
        if self.SMILES is None:
            await self.toSMILES()
            print(self.SMILES)
        folder_path = 'SMILES'
        subfolder = 'COMBINED_SMILES'
        if not os.path.exists(f'{folder_path}/{subfolder}'):
            os.makedirs(f'{folder_path}/{subfolder}')

        if os.path.exists(f'{folder_path}/{subfolder}/OUTPUT.json'):
            return json.loads(open(f'{folder_path}/{subfolder}/OUTPUT.json', 'r').read())

        pdf_canonical_smiles = []
        text_canonical_smiles = []
        for pdf_smiles in self.SMILES["PDF_SMILES"]:
            for keyword in pdf_smiles:
                SMILES = await fetch_from_pcp(keyword["SMILES"],"smiles", "canonical_smiles")
                
                pdf_canonical_smiles.append({
                    "SMILES": SMILES,
                    "cid": keyword["cid"],
                    "page": keyword["page"],
                    "X": keyword["X"],
                    "Y": keyword["Y"],
                    "origin": self.pdf_list[self.SMILES["PDF_SMILES"].index(pdf_smiles)].filename_without_extension
                })
            for text_smiles in self.SMILES["Text_SMILES"]:
                print(text_smiles)
                for keyword in text_smiles:
                    SMILES = await fetch_from_pcp(keyword["SMILES"],"smiles", "canonical_smiles")
                    
                    text_canonical_smiles.append({
                        "SMILES": SMILES,
                        "keyword": keyword["keyword"],
                        "cid": keyword["cid"],
                        "page": keyword["page"],
                        "index": keyword["index"],
                        "origin": self.text_list[self.SMILES["Text_SMILES"].index(text_smiles)].filename_without_extension
                    })
            final_list = []
            for pdf_smiles in pdf_canonical_smiles:
                for text_smiles in text_canonical_smiles:
                    if pdf_smiles["SMILES"] == text_smiles["SMILES"] and pdf_smiles["SMILES"] is not None:
                        final_list.append({
                            "SMILES": pdf_smiles["SMILES"],
                            "cid": pdf_smiles["cid"],
                            "page": pdf_smiles["page"],
                            "X": pdf_smiles["X"],
                            "Y": pdf_smiles["Y"],
                            "keyword": text_smiles["keyword"],
                            "index": text_smiles["index"],
                            "origin": pdf_smiles["origin"]
                        })
            

            for pdf_smiles in pdf_canonical_smiles:
                for smiles in final_list:
                    if pdf_smiles["SMILES"]  == smiles["SMILES"]:
                        pdf_canonical_smiles.remove(pdf_smiles)
                        break
            for text_smiles in text_canonical_smiles:
                for smiles in final_list:
                    if text_smiles["SMILES"]  == smiles["SMILES"]:
                        text_canonical_smiles.remove(text_smiles)
                        break
           
            for pdf_smiles in pdf_canonical_smiles:
                final_list.append({
                    "SMILES": pdf_smiles["SMILES"],
                    "cid": pdf_smiles["cid"],
                    "page": pdf_smiles["page"],
                    "X": pdf_smiles["X"],
                    "Y": pdf_smiles["Y"],
                    "origin": pdf_smiles["origin"]
                })


           
            for text_smiles in text_canonical_smiles:
                final_list.append({
                    "SMILES": text_smiles["SMILES"],
                    "cid": text_smiles["cid"],
                    "page": text_smiles["page"],
                    "keyword": text_smiles["keyword"],
                    "index": text_smiles["index"],
                    "origin": text_smiles["origin"]
                })

            
                        
       
            with open(f'{folder_path}/{subfolder}/OUTPUT.json', 'w') as f:
                json.dump(final_list, f)

            return final_list
        
                        
            
                        

                        





