"""Checks for the bundled model, request validation, and dataset paths."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

import app
from data_utils import load_combined_corpus


class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.load_model()

    def test_model_loads(self):
        self.assertTrue(app.health()['model_loaded'])
        self.assertEqual(type(app.pipeline.steps[-1][1]).__name__, 'LogisticRegression')

    def test_example_messages(self):
        examples = [
            ('WINNER!! You have been selected to receive a $1000 cash prize. Call now!', 'spam'),
            ('Hey, are we still on for dinner tonight at 7?', 'ham'),
        ]
        for message, expected in examples:
            with self.subTest(expected=expected):
                result = app.predict(app.PredictRequest(message=message))
                self.assertEqual(result.label, expected)
                self.assertEqual(result.is_spam, expected == 'spam')
                self.assertGreaterEqual(result.spam_probability, 0)
                self.assertLessEqual(result.spam_probability, 1)
                self.assertAlmostEqual(result.spam_probability + result.ham_probability, 1, places=3)


class InputAndDatasetTests(unittest.TestCase):
    def test_request_length_limits(self):
        for message in ('', 'x' * 2001):
            with self.assertRaises(ValidationError):
                app.PredictRequest(message=message)

    def test_whitespace_is_rejected(self):
        with self.assertRaises(HTTPException) as error:
            with patch.object(app, 'pipeline', object()):
                app.predict(app.PredictRequest(message='   '))
        self.assertEqual(error.exception.status_code, 400)

    def test_unloaded_model_returns_service_unavailable(self):
        with patch.object(app, 'pipeline', None):
            with self.assertRaises(HTTPException) as error:
                app.predict(app.PredictRequest(message='Hello'))
        self.assertEqual(error.exception.status_code, 503)

    def test_corpus_path_needs_no_trailing_slash(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'sample.csv').write_text('ham,Meet me tonight\nspam,Claim your prize\n', encoding='utf-8')
            with patch('data_utils.DATASET_FILES', ['sample.csv']):
                corpus = load_combined_corpus(folder)
        self.assertEqual(len(corpus), 2)
        self.assertEqual(set(corpus.label), {0, 1})


if __name__ == '__main__':
    unittest.main()
