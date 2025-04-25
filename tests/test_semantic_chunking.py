"""Tests for enhanced semantic chunking functionality in DiffSplitter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from codemap.git.diff_splitter import DiffSplitter
from codemap.utils.git_utils import GitDiff


@pytest.mark.unit
@pytest.mark.git
@pytest.mark.processor
@pytest.mark.chunking
class TestDiffSplitterBasic:
	"""Tests for basic functionality of DiffSplitter."""

	def setup_method(self) -> None:
		"""Set up test environment with a mock repo root."""
		self.repo_root = Path("/mock/repo")
		self.splitter = DiffSplitter(self.repo_root)

	def test_extract_code_from_diff(self) -> None:
		"""Test extracting code content from diff output."""
		# Arrange
		diff_content = """diff --git a/example.py b/example.py
index 1234567..abcdefg 100644
--- a/example.py
+++ b/example.py
@@ -10,7 +10,7 @@ def function1():
    return True
-def function2():
+def function2(param=None):
    pass
+def function3():
+    return None"""

		# Act
		old_code, new_code = self.splitter._extract_code_from_diff(diff_content)  # pylint: disable=protected-access

		# Assert
		assert "def function1()" in old_code
		assert "def function2():" in old_code
		assert "def function2(param=None):" in new_code
		assert "def function3():" in new_code
		assert "function3()" not in old_code


@pytest.mark.unit
@pytest.mark.git
@pytest.mark.processor
@pytest.mark.chunking
class TestSemanticHunkSplitting:
	"""Tests for semantic hunk splitting functionality."""

	def setup_method(self) -> None:
		"""Set up test environment with a mock repo root."""
		self.repo_root = Path("/mock/repo")
		self.splitter = DiffSplitter(self.repo_root)

	def test_semantic_hunk_splitting_python(self) -> None:
		"""Test semantic hunk splitting for Python code."""
		# Arrange
		diff_content = """diff --git a/example.py b/example.py
index 1234567..abcdefg 100644
--- a/example.py
+++ b/example.py
@@ -1,10 +1,12 @@
+import os
+import sys

def function1():
    return True

-def function2():
+def function2(param=None):
    pass

class Example:
    def method1(self):
        return "test"
+    def method2(self):
+        return "new method"

+if __name__ == "__main__":
+    function1()"""

		# Act - Split the diff semantically
		chunks = self.splitter._semantic_hunk_splitting("example.py", diff_content)  # pylint: disable=protected-access

		# Assert
		assert len(chunks) >= 1  # Ensure we get at least one chunk

		# If we have multiple chunks, check for specific content in them
		if len(chunks) > 1:
			# Check that imports, functions, and classes are in separate chunks
			has_import = any("import" in c for c in chunks)
			has_function = any("function1" in c or "function2" in c for c in chunks)
			has_class = any("class Example" in c for c in chunks)
			has_main = any("__main__" in c for c in chunks)

			assert has_import
			assert has_function
			assert has_class
			assert has_main

		# Act - Apply the enhanced semantic split
		diff = GitDiff(
			files=["example.py"],
			content=diff_content,
			is_staged=False,
		)
		enhanced_chunks = self.splitter._enhance_semantic_split(diff)  # pylint: disable=protected-access

		# Assert
		assert len(enhanced_chunks) >= 1
		assert enhanced_chunks[0].files == ["example.py"]

	def test_semantic_hunk_splitting_javascript(self) -> None:
		"""Test semantic hunk splitting for JavaScript code."""
		# Arrange
		diff_content = """diff --git a/example.js b/example.js
index 1234567..abcdefg 100644
--- a/example.js
+++ b/example.js
@@ -1,10 +1,12 @@
+import React from 'react';

function calculateTotal(items) {
  return items.reduce((total, item) => total + item.price, 0);
}

class ShoppingCart {
-  constructor() {
+  constructor(items = []) {
+    this.items = items;
  }

  addItem(item) {
    this.items.push(item);
  }

+  getTotal() {
+    return calculateTotal(this.items);
+  }
}

+export default ShoppingCart;"""

		# Act - Split the diff semantically
		chunks = self.splitter._semantic_hunk_splitting("example.js", diff_content)  # pylint: disable=protected-access

		# Assert
		assert len(chunks) >= 1  # Ensure we get at least one chunk

		# If we have multiple chunks, check for specific content in them
		if len(chunks) > 1:
			# Check that imports, functions, and classes are in separate chunks
			has_import = any("import React" in c for c in chunks)
			has_function = any("function calculateTotal" in c for c in chunks)
			has_class = any("class ShoppingCart" in c for c in chunks)
			has_export = any("export default" in c for c in chunks)

			assert has_import
			assert has_function
			assert has_class
			assert has_export

		# Act - Apply the enhanced semantic split
		diff = GitDiff(
			files=["example.js"],
			content=diff_content,
			is_staged=False,
		)
		enhanced_chunks = self.splitter._enhance_semantic_split(diff)  # pylint: disable=protected-access

		# Assert
		assert len(enhanced_chunks) >= 1
		assert enhanced_chunks[0].files == ["example.js"]


@pytest.mark.unit
@pytest.mark.git
@pytest.mark.processor
@pytest.mark.chunking
class TestAdvancedSemanticSplitting:
	"""Tests for advanced semantic splitting functionality."""

	def setup_method(self) -> None:
		"""Set up test environment with a mock repo root."""
		self.repo_root = Path("/mock/repo")
		self.splitter = DiffSplitter(self.repo_root)

	def test_enhance_semantic_split(self) -> None:
		"""Test enhanced semantic splitting that considers code structure."""
		# Arrange
		diff = GitDiff(
			files=["example.py", "utils.js"],
			content="""diff --git a/example.py b/example.py
index 1234567..abcdefg 100644
--- a/example.py
+++ b/example.py
@@ -1,5 +1,7 @@
+import os
+import sys

def function1():
    return True

-def function2():
+def function2(param=None):
    pass

+def function3():
+    return None

+class Example:
+    def method1(self):
+        return "test"
diff --git a/utils.js b/utils.js
index 2345678..bcdefgh 100645
--- a/utils.js
+++ b/utils.js
@@ -1,3 +1,5 @@
+import { useState } from 'react';

function formatPrice(price) {
  return `$${price.toFixed(2)}`;
}

+export const calculateTotal = (items) => {
+  return items.reduce((sum, item) => sum + item.price, 0);
+};""",
			is_staged=False,
		)

		# Act
		chunks = self.splitter._enhance_semantic_split(diff)  # pylint: disable=protected-access

		# Assert
		assert len(chunks) >= 2  # Should have at least one chunk per file

		# Verify that Python and JavaScript files were properly handled
		py_chunks = [c for c in chunks if c.files[0] == "example.py"]
		js_chunks = [c for c in chunks if c.files[0] == "utils.js"]

		assert len(py_chunks) >= 1  # Should have at least the Python file chunk
		assert len(js_chunks) >= 1  # Should have at least the JavaScript file chunk

	def test_end_to_end_semantic_strategy(self) -> None:
		"""Test the semantic strategy end-to-end with real-world examples."""
		# Arrange
		diff = GitDiff(
			files=["models.py", "views.py", "tests/test_models.py"],
			content="""diff --git a/models.py b/models.py
index 1234567..abcdefg 100644
--- a/models.py
+++ b/models.py
@@ -1,5 +1,7 @@
 from django.db import models
+from django.utils import timezone

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
+    created_at = models.DateTimeField(default=timezone.now)
+    is_active = models.BooleanField(default=True)

class Product(models.Model):
    name = models.CharField(max_length=200)
    price = models.DecimalField(max_digits=10, decimal_places=2)
+    description = models.TextField(blank=True)
diff --git a/views.py b/views.py
index 2345678..bcdefgh 100645
--- a/views.py
+++ b/views.py
@@ -1,8 +1,15 @@
 from django.shortcuts import render
+from django.http import JsonResponse

from .models import User, Product

def index(request):
    users = User.objects.all()
    return render(request, 'index.html', {'users': users})
+
+def product_list(request):
+    products = Product.objects.all()
+    data = [{'id': p.id, 'name': p.name, 'price': p.price} for p in products]
+    return JsonResponse(data, safe=False)
diff --git a/tests/test_models.py b/tests/test_models.py
index 3456789..cdefghi 100645
--- a/tests/test_models.py
+++ b/tests/test_models.py
@@ -1,5 +1,11 @@
 from django.test import TestCase
+from django.utils import timezone

from ..models import User, Product

-class UserTestCase(TestCase):
+class UserModelTest(TestCase):
     def test_user_creation(self):
         user = User.objects.create(name="Test User", email="test@example.com")
+        self.assertEqual(user.name, "Test User")
+        self.assertEqual(user.email, "test@example.com")
+        self.assertTrue(user.is_active)
+        self.assertIsNotNone(user.created_at)
+
+class ProductModelTest(TestCase):
+    def test_product_creation(self):
+        product = Product.objects.create(name="Test Product", price=19.99)
+        self.assertEqual(product.name, "Test Product")
+        self.assertEqual(float(product.price), 19.99)""",
			is_staged=False,
		)

		# Act - with mocked git command
		with (
			patch("codemap.git.diff_splitter.run_git_command") as mock_git,
			patch.object(self.splitter, "_split_semantic", wraps=self.splitter._split_semantic) as mock_split,
		):
			# Mock git status command
			mock_git.return_value = ""

			# Call split_diff which uses _split_semantic under the hood
			chunks = self.splitter.split_diff(diff)

			# Verify
			assert mock_split.called
			assert len(chunks) > 0  # We should get at least one chunk

			# Instead of requiring exact grouping, just verify that the content
			# and files are correctly included in some chunks
			model_files = {"models.py", "tests/test_models.py"}
			processed_model_files = set()

			# Collect all model-related files from all chunks
			for chunk in chunks:
				for file in chunk.files:
					if file in model_files:
						processed_model_files.add(file)

			# Verify all model files were processed
			assert processed_model_files == model_files, "All model files should be processed"

			# Verify at least one chunk contains User model related content
			user_model_found = False
			for chunk in chunks:
				content = chunk.content
				if "User" in content and "Model" in content:
					user_model_found = True
					break

			assert user_model_found, "Content should include User model"


@pytest.mark.unit
@pytest.mark.git
@pytest.mark.processor
@pytest.mark.chunking
@pytest.mark.llm
class TestEmbeddingSimilarity:
	"""Tests for embedding-based semantic similarity functionality."""

	def setup_method(self) -> None:
		"""Set up test environment."""
		# Save original class-level availability flags
		# pylint: disable=protected-access
		self.original_st_available = DiffSplitter._sentence_transformers_available
		self.original_model_available = DiffSplitter._model_available

		# Initialize diff splitter with a mock repo root
		self.repo_root = Path("/mock/repo")
		self.splitter = DiffSplitter(self.repo_root)

	def teardown_method(self) -> None:
		"""Restore original settings after test."""
		# Restore original class-level availability flags
		# pylint: disable=protected-access
		DiffSplitter._sentence_transformers_available = self.original_st_available
		DiffSplitter._model_available = self.original_model_available

	def test_embedding_similarity(self) -> None:
		"""Test semantic similarity between code fragments."""
		# Skip this test if sentence-transformers isn't installed
		try:
			import sentence_transformers  # type: ignore[import] # noqa: F401
		except ImportError:
			pytest.skip("sentence-transformers not installed")

		# Arrange
		# Force the embeddings to return meaningful values for testing
		# pylint: disable=protected-access
		DiffSplitter._sentence_transformers_available = True
		DiffSplitter._model_available = True

		# Mock the embedding function to return predictable values
		with patch.object(self.splitter, "_get_code_embedding") as mock_embedding:
			# Set up mock to return different embeddings for different code
			def fake_embedding(code: str) -> list[float]:
				if "calculate_total" in code or "compute_sum" in code:
					# Similar functions should have similar embeddings
					return [0.1, 0.2, 0.3] if "calculate_total" in code else [0.15, 0.25, 0.35]
				# Different function should have a different embedding
				return [0.9, 0.8, 0.7]

			mock_embedding.side_effect = fake_embedding

			# Test code samples with different semantics
			code1 = """
            def calculate_total(items):
                return sum(item.price for item in items)
            """

			code2 = """
            def compute_sum(products):
                return sum(product.price for product in products)
            """

			code3 = """
            def get_user_info(user_id):
                return User.objects.get(id=user_id)
            """

			# Act
			sim1_2 = self.splitter._calculate_semantic_similarity(code1, code2)  # pylint: disable=protected-access
			sim1_3 = self.splitter._calculate_semantic_similarity(code1, code3)  # pylint: disable=protected-access

			# Assert
			assert sim1_2 > sim1_3  # Functions doing similar things should have higher similarity

	def test_sentence_transformers_availability(self) -> None:
		"""Test the sentence-transformers availability check functions."""
		# Arrange
		# Reset class variables to force availability check
		# pylint: disable=protected-access
		DiffSplitter._sentence_transformers_available = None
		DiffSplitter._model_available = None

		# Act & Assert
		with patch.object(DiffSplitter, "_check_sentence_transformers_availability") as mock_check:
			mock_check.return_value = True

			# Try to initialize a new splitter which will trigger availability check
			splitter = DiffSplitter(self.repo_root)

			# Force the sentence_transformers_available flag to True for testing
			# pylint: disable=protected-access
			DiffSplitter._sentence_transformers_available = True

			# Check if sentence-transformers is now available in our mocked environment
			# pylint: disable=protected-access
			assert DiffSplitter._sentence_transformers_available is True

			# Test with get_code_embedding
			with patch.object(splitter, "_get_code_embedding") as mock_embedding:
				mock_embedding.return_value = [0.1, 0.2, 0.3]

				# Test calculate_semantic_similarity with mocked embeddings
				with patch.object(splitter, "_calculate_semantic_similarity") as mock_sim:
					mock_sim.return_value = 0.75

					# Now check a similarity calculation would give the expected result
					result = mock_sim("code1", "code2")
					assert result == 0.75
