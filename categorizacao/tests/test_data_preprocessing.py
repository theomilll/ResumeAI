import os
import sys
import unittest

import pandas as pd
import torch

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import (SummaryDataset, load_and_preprocess_data,
                                    prepare_datasets, save_label_encoder)


class TestDataPreprocessing(unittest.TestCase):
    """Tests for data preprocessing module."""
    
    def setUp(self):
        """Set up test data."""
        # Create test data
        self.test_data = pd.DataFrame({
            'text': [
                "Discutimos o progresso do projeto X e definimos os próximos passos.",
                "O departamento de pesquisa apresentou os resultados do estudo de usabilidade.",
                "Realizamos a distribuição de tarefas para a próxima semana.",
                "Apresentamos a proposta comercial para o cliente XYZ.",
                "Discutimos questões administrativas e procedimentos internos."
            ],
            'category': [
                "Atualizações de Projeto",
                "Achados de Pesquisa",
                "Gestão de Equipe",
                "Reuniões com Clientes",
                "Outros"
            ]
        })
        
        # Save test data to a temporary CSV file
        os.makedirs('tests/temp', exist_ok=True)
        self.test_data_path = 'tests/temp/test_data.csv'
        self.test_data.to_csv(self.test_data_path, index=False)
        
    def tearDown(self):
        """Clean up test data."""
        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)
            
        if os.path.exists('tests/temp/label_encoder.csv'):
            os.remove('tests/temp/label_encoder.csv')
            
        if os.path.exists('tests/temp'):
            os.rmdir('tests/temp')
    
    def test_load_and_preprocess_data(self):
        """Test loading and preprocessing data."""
        train_texts, val_texts, train_labels, val_labels, label_encoder = load_and_preprocess_data(
            self.test_data_path, test_size=0.4, random_state=42
        )
        
        # Check that the data was split correctly
        self.assertEqual(len(train_texts), 3)
        self.assertEqual(len(val_texts), 2)
        self.assertEqual(len(train_labels), 3)
        self.assertEqual(len(val_labels), 2)
        
        # Check that the label encoder contains all categories
        self.assertEqual(len(label_encoder), 5)
        self.assertIn("Atualizações de Projeto", label_encoder)
        self.assertIn("Achados de Pesquisa", label_encoder)
        self.assertIn("Gestão de Equipe", label_encoder)
        self.assertIn("Reuniões com Clientes", label_encoder)
        self.assertIn("Outros", label_encoder)
    
    def test_prepare_datasets(self):
        """Test preparing datasets."""
        train_texts, val_texts, train_labels, val_labels, _ = load_and_preprocess_data(
            self.test_data_path, test_size=0.4, random_state=42
        )
        
        train_dataset, val_dataset, tokenizer = prepare_datasets(
            train_texts, val_texts, train_labels, val_labels
        )
        
        # Check that the datasets have the correct size
        self.assertEqual(len(train_dataset), 3)
        self.assertEqual(len(val_dataset), 2)
        
        # Check that the tokenizer was loaded
        self.assertIsNotNone(tokenizer)
        
        # Check that the dataset items have the expected format
        item = train_dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('label', item)
        
        # Check that the tensors have the correct shape
        self.assertEqual(item['input_ids'].dim(), 1)
        self.assertEqual(item['attention_mask'].dim(), 1)
        self.assertEqual(item['label'].dim(), 0)
        
        # Check that the label is a tensor
        self.assertIsInstance(item['label'], torch.Tensor)
    
    def test_save_label_encoder(self):
        """Test saving label encoder."""
        label_encoder = {
            "Atualizações de Projeto": 0,
            "Achados de Pesquisa": 1,
            "Gestão de Equipe": 2,
            "Reuniões com Clientes": 3,
            "Outros": 4
        }
        
        encoder_path = save_label_encoder(label_encoder, 'tests/temp')
        
        # Check that the encoder was saved
        self.assertTrue(os.path.exists(encoder_path))
        
        # Check that the encoder has the correct format
        encoder_df = pd.read_csv(encoder_path)
        self.assertEqual(len(encoder_df), 5)
        self.assertIn('idx', encoder_df.columns)
        self.assertIn('category', encoder_df.columns)
        
        # Check that all categories are in the encoder
        categories = encoder_df['category'].tolist()
        self.assertIn("Atualizações de Projeto", categories)
        self.assertIn("Achados de Pesquisa", categories)
        self.assertIn("Gestão de Equipe", categories)
        self.assertIn("Reuniões com Clientes", categories)
        self.assertIn("Outros", categories)

if __name__ == '__main__':
    unittest.main() 