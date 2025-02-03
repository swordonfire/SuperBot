# src/services/telegram_service.py
import logging
import os
from pathlib import Path

from telegram import Document, Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from src.core.config.settings import settings
from src.llm.engine import get_llm_engine
from src.rag import Chunker, Embedder, RAGPipeline, Retriever, VectorStore

logger = logging.getLogger(__name__)


class TelegramBotService:
    def __init__(self):
        # Initialize RAG components
        self.chunker = Chunker(chunk_size=512, chunk_overlap=64)
        self.embedder = Embedder()
        self.vector_store = VectorStore(persist_dir='data/vector_db/chroma')
        self.retriever = Retriever(
            vector_store=self.vector_store, embedder=self.embedder, hybrid_search=True
        )
        self.rag_pipeline = RAGPipeline(self.retriever)

        # Initialize LLM
        self.llm = get_llm_engine()

        # Initialize Telegram bot
        self.app = (
            ApplicationBuilder()
            .token(settings.TELEGRAM_BOT_TOKEN)
            .post_init(self._register_commands)
            .build()
        )

        self._setup_handlers()

    def _setup_handlers(self):
        """Configure bot command and message handlers"""
        self.app.add_handler(CommandHandler('start', self._handle_start))
        self.app.add_handler(CommandHandler('help', self._handle_help))
        self.app.add_handler(MessageHandler(filters.Document.PDF, self._handle_document))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_query))

    async def _register_commands(self, app):
        """Register bot commands menu"""
        await app.bot.set_my_commands(
            [
                ('start', 'Initialize the bot'),
                ('help', 'Show help information'),
                ('upload', 'Upload a PDF document'),
                ('tweet_help', 'Generate and iterate tweet ideas'),
                ('mode', 'Select Chat Mode'),
                ('find_member', 'Find the members based on you requirements')
            ]
        )

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Welcome message with bot capabilities"""
        user = update.effective_user
        welcome_msg = (
            f"👋 Welcome {user.first_name} to Superteam Vietnam's AI Assistant!\n\n"
            '=> Capabilities: \n'
            '• Upload PDF documents with /upload\n'
            '• Ask questions about our community\n'
            '• Find relevant resources and members\n\n'
            'Try sending a PDF or asking a question!'
        )
        await update.message.reply_text(welcome_msg)

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Detailed help instructions"""
        help_text = (
            '📚 Superteam Vietnam AI Assistant Help\n\n'
            '1. Upload Documents\n'
            '   Send a PDF file or use /upload\n\n'
            '2. Ask Questions\n'
            '   Type your question normally\n\n'
            '3. Member Search\n'
            '   Try: "Find a Rust developer with DeFi experience"\n\n'
            '4. Content Assistance\n'
            '   Request tweet drafts or message improvements'
        )
        await update.message.reply_text(help_text)

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Process PDF uploads with progress updates"""
        try:
            doc: Document = update.message.document
            if doc.mime_type != 'application/pdf':
                await update.message.reply_text('❌ Please send a valid PDF file')
                return

            await update.message.reply_text('📥 Starting document processing...')

            # Download file
            file_path = await self._download_pdf(update, doc)

            # Process document through RAG pipeline
            chunks = self.chunker.chunk_document(file_path)
            embeddings = self.embedder.embed_chunks(chunks)
            self.vector_store.add_documents(chunks, embeddings)

            # Cleanup and confirmation
            os.remove(file_path)
            await update.message.reply_text(
                f'✅ Successfully processed {len(chunks)} knowledge chunks!\nFile: {doc.file_name}'
            )

        except Exception as e:
            logger.error(f'Document processing failed: {str(e)}')
            await update.message.reply_text('❌ Failed to process document. Please try again.')

    async def _handle_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle user queries with RAG and LLM"""
        try:
            query = update.message.text.strip()
            if not query:
                return

            # Show typing indicator
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')

            # Process query through RAG pipeline
            response = self.rag_pipeline.generate_response(query)

            # Format response for Telegram
            formatted_response = self._format_response(response, query)

            await update.message.reply_text(formatted_response)

        except Exception as e:
            logger.error(f'Query handling failed: {str(e)}')
            await update.message.reply_text('⚠️ Sorry, I encountered an error. Please try again.')

    async def _download_pdf(self, update: Update, doc: Document) -> str:
        """Download PDF to local storage"""
        os.makedirs('data/documents', exist_ok=True)
        file_path = Path('data/documents') / doc.file_name

        # Handle duplicate filenames
        counter = 1
        while file_path.exists():
            file_path = Path('data/documents') / f'{counter}_{doc.file_name}'
            counter += 1

        # Download the file
        file = await doc.get_file()
        await file.download_to_drive(file_path)
        return str(file_path)

    def _format_response(self, response: str, query: str) -> str:
        """Format LLM response for Telegram"""
        # Truncate long responses
        MAX_LENGTH = 1500
        if len(response) > MAX_LENGTH:
            response = response[:MAX_LENGTH] + '...\n\n(Message truncated)'

        # Add query context
        return f'🔍 **Query:** {query}\n\n{response}'

    def run(self):
        """Start the Telegram bot"""
        logger.info('Starting Telegram bot...')
        self.app.run_polling(poll_interval=1, timeout=30, drop_pending_updates=True)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
    )
    TelegramBotService().run()
