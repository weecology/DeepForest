import gradio as gr
from deepforest import main
import cv2


def show_trees(img_path):
    model = main.deepforest()
    model.use_release()
    img = model.predict_image(path=img_path, return_plot=True)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_birds(img_path):
    model = main.deepforest()
    model.use_bird_release()
    img = model.predict_image(path=img_path, return_plot=True)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

with gr.Blocks() as demo:
    gr.Markdown('# Deepforest')
    gr.Markdown('## Tree Detection Model')
    gr.Markdown('### Predict trees')
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="filepath")
        with gr.Column():
            output_image = gr.Image(label="Predicted image")

    submit_button_trees = gr.Button("Predict trees")
    submit_button_trees.click(show_trees, inputs=input_image, outputs=output_image)
    
    gr.Markdown('### Predict birds')
    with gr.Row():
        with gr.Column():
            input_image=gr.Image(label="Input image",type="filepath")
        with gr.Column():
            output_image=gr.Image(label="Predicted Image")
            
    submit_button_birds = gr.Button("Predict birds")
    submit_button_birds.click(show_birds,inputs=input_image,outputs=output_image)

if __name__ == "__main__":
    demo.launch()