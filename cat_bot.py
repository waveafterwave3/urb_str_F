##BOT V TG(@MODELYRBOT)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.preprocessing import LabelEncoder
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import pickle

tokenizer = AutoTokenizer.from_pretrained('./model9ep')
model = AutoModelForSequenceClassification.from_pretrained('./model9ep')

with open('./model9ep/label_classes.pkl', 'rb') as f:
    label_encoder = LabelEncoder()
    label_encoder.classes_ = pickle.load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = np.argmax(logits.cpu().numpy(), axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
    return predicted_label

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('123')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    category = classify_text(user_message)
    await update.message.reply_text(f'Это сообщение классифицировано как: {category}')

def main():
    app = ApplicationBuilder().token("7694723703:AAHnO_0xjTCY0k-5eBqgfwf4qGosJ123ngs").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()

if __name__ == '__main__':
    main()
