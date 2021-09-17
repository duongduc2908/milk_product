import json 

from PIL import Image
from matplotlib import pyplot as plt
import re
import ngram

class Analyzer:
    brands = {
        'abbott', 'bellamyorganic', 'blackmores', 'bubs_australia', 'danone', 'f99foods', 'friesland_campina_dutch_lady',
        'gerber', 'glico', 'heinz', 'hipp', 'humana uc', 'mead_johnson', 'megmilksnowbrand', 'meiji', 'morigana',
        'namyang', 'nestle', 'no_brand', 'nutifood', 'nutricare', 'pigeon', 'royal_ausnz', 'vinamilk',
        'vitadairy', 'wakodo'
    }

    def __init__(self, n=3,):
        self.n = n
        self.index = ngram.NGram(N=n)

    def __call__(self, s):
        tokens = re.split(r'\s+', s.lower().strip())
        filtered_tokens = []
        for token in tokens:
            if len(token) > 20:
                continue

            if re.search(r'[?\[\]\(\):!]', token):
                continue

            if re.search(f'\d{2,}', token):
                continue

            filtered_tokens.append(token)

        non_ngram_tokens = []
        ngram_tokens = []

        for token in filtered_tokens:
            if token in self.brands:
                non_ngram_tokens.append(token)
                n_grams = list(self.index.ngrams(self.index.pad(token)))
                ngram_tokens.extend(n_grams)
            else:
                n_grams = list(self.index.ngrams(self.index.pad(token)))
                ngram_tokens.extend(n_grams)
        res = [*non_ngram_tokens, *ngram_tokens]
        return res

def display_image(im_cv):
  im_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
  pil_img = Image.fromarray(im_cv)
  plt.imshow(pil_img)
  plt.show()

def meger_label_branch(labels,step,name):
    for line in labels:
        words = line.split("/")
        if words[step] == name.lower().replace(" ","_"):
            return words[0]

def word2line(result, img):
    temp = {'center': None, 'text': None}
    new_res = []
    zero_mask = np.zeros(img.shape[:2]).astype('uint8')
    zero_mask_copy = zero_mask.copy()
    for res in result:
        x,y,w,h = cv2.boundingRect(res['boxes'].astype(int))
        zero_mask[y+int(0.3*h):y+int(0.7*h), x:x+w] = 125
        # zero_mask = cv2.polylines(zero_mask, [res['boxes'].astype(int)], True, 255, -1)

        center = np.array([x+0.5*w, y+0.5*h]).astype(int)
        # print(cv2.pointPolygonTest(res['boxes'].astype(int),tuple(center),False))
        item = temp.copy()
        item['center'] = center
        item['text'] = res['text']
        new_res.append(item)

    kernel = np.ones((1, 20), np.uint8)    
    zero_mask = cv2.dilate(zero_mask, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(zero_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # zero_mask_copy = cv2.drawContours(zero_mask_copy, contours, -1, 255, 2)
    # cv2.imrite('mask.jpg', zero_mask_copy)

    temp = {'contour': None, 'text': None, 'box': None}
    final_res = []  
    for contour in contours:
        box = cv2.boundingRect(contour.astype(int))
        item = temp.copy()
        item['box'] = np.array(box)
        item['contour'] = contour

        text_with_center = []
        temp1 = {'center': None, 'text': None}
        for pt in new_res:
            if cv2.pointPolygonTest(contour,tuple(pt['center']),False) > 0:
                item1 = temp1.copy()
                item1['text'] = pt['text']
                item1['center'] = pt['center']
                text_with_center.append(item1)
        
        text_with_center = np.array(text_with_center)
        only_center = [it['center'][0] for it in text_with_center]
        text_with_center = text_with_center[np.argsort(only_center)]
        
        item['text'] = ' '.join([text['text'] for text in text_with_center])
        final_res.append(item)

    return final_res

class Ensemble():
    
    def __init__(self,branch,result_chinh,result_thanh, json_chinh, json_thanh, text_list):
        self.branch = branch 
        self.result_chinh = result_chinh
        self.result_thanh = result_thanh
        self.text_list = text_list
        self.json_chinh_dict = json_chinh
        self.json_thanh_dict = json_thanh
        # with open(json_chinh,'r') as f:
        #     self.json_chinh_dict =json.load(f)
        # with open(json_thanh,'r') as f:
        #     self.json_thanh_dict =json.load(f)
    
    def run(self,):
        

        result_chinh = self.json_chinh_dict[str(self.result_chinh)]
        if len(result_chinh)>2:
            check_list = True
        else:
            check_list = False
        if len(result_chinh[-1].split('/'))==3:
          branch_chinh,middle_chinh,step_chinh = result_chinh[-1].split('/')
        else:
          branch_chinh,middle_chinh = result_chinh[-1].split('/')
          step_chinh = middle_chinh
        age = -1
        for text in self.text_list:
            if text.isnumeric():
                if int(text)< 10:
                    if age == -1:
                        age = int(text)
                    if age > int(text):
                        age = int(text)
        if age >= 0:
            age = str(age)
        else:
            age = ''
        if self.result_thanh is None:
          if branch_chinh == self.branch:
            if age!='':
              label = self.branch +'/'+middle_chinh +'/'+age
            else:
              label = self.branch +'/' + middle_chinh
            return label
          else:
            return self.branch



        if branch_chinh == self.branch:
            if  age!='': # có tuổi
                label = self.branch + '/' + middle_chinh + '/' + age

            
            else: #không có tuổi
                if not check_list:
                    label = self.branch +'/' + middle_chinh + '/' + step_chinh 
                else: 
                    if self.result_thanh in [i.split('/')[-1] for i in result_chinh]:
                        label = self.branch + '/' + middle_chinh 
                        if middle_chinh=='dutch_baby':
                          label = self.branch + '/' + middle_chinh +'/' + self.result_thanh

                    else:
                        label = self.branch + '/' + middle_chinh 
                    if self.result_thanh=='other':
                        label = self.branch
                    

        else:  # chính khác branch gốc 
            
            if self.result_thanh == 'other':
                label = self.branch 
            else:
                middle_thanh =  self.json_thanh_dict[self.result_thanh].split('/')[0]
                if  age!='': # có tuổi
                    label = self.branch + '/' + middle_thanh + '/' + age
                else: 
                    label = self.branch + '/' + middle_thanh
        if 'nan_optipro' in [i.strip() for i in label.split("/")]:
            if 'optipro' not in [t.lower() for t in self.text_list]:
                label=label.replace('nan_optipro','nan')
        # if 'nan' in [i.strip() for i in label.split("/")]:
        #     if 'optipro' in [t.lower() for t in self.text_list]:
        #         label=label.replace('nan','nan_optipro')
        if 'dielac_grow_plus_blue' in [i.strip() for i in label.split("/")] or 'dielac_grow_plus_red' in [i.strip() for i in label.split("/")]:
            if 'plus' not in self.text_list:
                  label=label.replace('dielac_grow_plus_blue','dielac_grow_blue').replace('dielac_grow_plus_red','dielac_grow_red')
        if 'oggi' in self.text_list or '0ggi' in self.text_list or 'Ogi' in self.text_list :
            label = "vitadary/oggi/" + age
        return label