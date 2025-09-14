import streamlit as st
import torch
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

st.set_page_config(
    page_title='VQA Demo',
)

@st.cache_resource
def load_model():
    local_model_path = r'D:\Final\paligemma_vqav2_10pc'

    device = 'cpu'  
    st.info('Sử dụng CPU để tải model.')

    processor = PaliGemmaProcessor.from_pretrained(local_model_path)
    st.success('Tải processor từ model local')

    model_kwargs = {
        'torch_dtype': torch.float32,  
        'low_cpu_mem_usage': True,
    }

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        local_model_path,
        **model_kwargs
    )

    st.success(f'Model đã được tải trên: {device.upper()}')

    return processor, model

def process_image(image: Image.Image) -> Image.Image:

    if image.mode != 'RGB':
        image = image.convert('RGB')

    max_size = 448
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image

def generate_answer(processor, model, image: Image.Image, question: str) -> str:

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    inputs = processor(
        text=question,
        images=image,
        return_tensors='pt'
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,  
            do_sample=False,
            num_beams=1,  
            pad_token_id=processor.tokenizer.eos_token_id
        )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    if question.lower() in generated_text.lower():
        answer = generated_text.lower().replace(question.lower(), '').strip()
    else:
        answer = generated_text.strip()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return answer if answer else 'Không thể tạo câu trả lời.'


def main():
    st.title('Visual Question Answering Demo')
    with st.spinner('Đang tải model...'):
        processor, model = load_model()

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header('Tải ảnh')

        uploaded_file = st.file_uploader(
            'Chọn ảnh', 
            type=['png', 'jpg', 'jpeg'],
            help='Hỗ trợ định dạng: PNG, JPG, JPEG'
        )
        image = None
        if uploaded_file:
            image = Image.open(uploaded_file)
        if image:
            image = process_image(image)
            st.image(image, caption='Ảnh đã tải', use_container_width=True)
    
    with col2:
        st.header('Đặt câu hỏi')

        question = st.text_input(
            'Nhập câu hỏi về ảnh:',
            placeholder='Ví dụ: What is in the image?'
        )
        if st.button('Trả lời', type='primary', use_container_width=True):
            if question.strip():
                with st.spinner('Đang phân tích ảnh và tạo câu trả lời...'):
                    answer = generate_answer(processor, model, image, question)
                
                st.subheader('Câu trả lời:')
                st.write(f'**Câu hỏi:** {question}')
                st.write(f'**Trả lời:** {answer}')
            else:
                st.warning('Vui lòng nhập câu hỏi!')

if __name__ == '__main__':
    main()