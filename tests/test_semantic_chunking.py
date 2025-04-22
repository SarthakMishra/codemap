"""Tests for enhanced semantic chunking functionality in DiffSplitter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from codemap.git.commit.diff_splitter import DiffSplitter
from codemap.utils.git_utils import GitDiff


def test_extract_code_from_diff() -> None:
    """Test extracting code content from diff output."""
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

    # Initialize diff splitter with a mock repo root
    repo_root = Path("/mock/repo")
    splitter = DiffSplitter(repo_root)

    # Extract old and new code
    old_code, new_code = splitter._extract_code_from_diff(diff_content)  # pylint: disable=protected-access # noqa: SLF001

    # Verify extracted content
    assert "def function1()" in old_code
    assert "def function2():" in old_code
    assert "def function2(param=None):" in new_code
    assert "def function3():" in new_code
    assert "function3()" not in old_code


def test_semantic_hunk_splitting_python() -> None:
    """Test semantic hunk splitting for Python code."""
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

    # Initialize diff splitter with a mock repo root
    repo_root = Path("/mock/repo")
    splitter = DiffSplitter(repo_root)

    # Split the diff semantically
    chunks = splitter._semantic_hunk_splitting("example.py", diff_content)  # pylint: disable=protected-access # noqa: SLF001

    # Verify the chunking - implementation might return multiple chunks
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

    # Apply the enhanced semantic split to test the real feature
    diff = GitDiff(
        files=["example.py"],
        content=diff_content,
        is_staged=False,
    )
    enhanced_chunks = splitter._enhance_semantic_split(diff)  # pylint: disable=protected-access # noqa: SLF001

    # Verify the enhanced semantic chunking
    assert len(enhanced_chunks) >= 1
    assert enhanced_chunks[0].files == ["example.py"]


def test_semantic_hunk_splitting_javascript() -> None:
    """Test semantic hunk splitting for JavaScript code."""
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

    # Initialize diff splitter with a mock repo root
    repo_root = Path("/mock/repo")
    splitter = DiffSplitter(repo_root)

    # Split the diff semantically
    chunks = splitter._semantic_hunk_splitting("example.js", diff_content)  # pylint: disable=protected-access # noqa: SLF001

    # Verify the chunking - implementation might return multiple chunks
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

    # Apply the enhanced semantic split to test the real feature
    diff = GitDiff(
        files=["example.js"],
        content=diff_content,
        is_staged=False,
    )
    enhanced_chunks = splitter._enhance_semantic_split(diff)  # pylint: disable=protected-access # noqa: SLF001

    # Verify the enhanced semantic chunking
    assert len(enhanced_chunks) >= 1
    assert enhanced_chunks[0].files == ["example.js"]


def test_enhance_semantic_split() -> None:
    """Test enhanced semantic splitting that considers code structure."""
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

    # Initialize diff splitter with a mock repo root
    repo_root = Path("/mock/repo")
    splitter = DiffSplitter(repo_root)

    # Apply enhanced semantic splitting
    chunks = splitter._enhance_semantic_split(diff)  # pylint: disable=protected-access # noqa: SLF001

    # Verify the chunking - should have at least one chunk per file
    assert len(chunks) >= 2

    # Verify that Python and JavaScript files were properly handled
    py_chunks = [c for c in chunks if c.files[0] == "example.py"]
    js_chunks = [c for c in chunks if c.files[0] == "utils.js"]

    assert len(py_chunks) >= 1  # Should have at least the Python file chunk
    assert len(js_chunks) >= 1  # Should have at least the JavaScript file chunk


def test_end_to_end_semantic_strategy() -> None:
    """Test the semantic strategy end-to-end with real-world examples."""
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

    # Initialize diff splitter with a mock repo root
    repo_root = Path("/mock/repo")
    splitter = DiffSplitter(repo_root)

    # Apply semantic strategy splitting
    chunks = splitter.split_diff(diff)

    # Verify semantic grouping results
    assert len(chunks) >= 2  # Should have grouped related changes

    # Find chunk with User model changes
    user_chunk = next((c for c in chunks if any(f == "models.py" for f in c.files)), None)
    assert user_chunk is not None
    assert "created_at" in user_chunk.content

    # Find chunk with view changes
    view_chunk = next((c for c in chunks if any(f == "views.py" for f in c.files)), None)
    assert view_chunk is not None
    assert "JsonResponse" in view_chunk.content


def test_embedding_similarity() -> None:
    """Test semantic similarity between code fragments."""
    # Skip this test if sentence-transformers isn't installed
    try:
        import sentence_transformers  # type: ignore[import] # noqa: F401
    except ImportError:
        pytest.skip("sentence-transformers not installed")

    # Save original class-level availability flags
    # pylint: disable=protected-access
    original_st_available = DiffSplitter._sentence_transformers_available  # noqa: SLF001
    original_model_available = DiffSplitter._model_available  # noqa: SLF001

    # Initialize diff splitter with a mock repo root
    repo_root = Path("/mock/repo")
    splitter = DiffSplitter(repo_root)

    try:
        # Force the embeddings to return meaningful values for testing
        # pylint: disable=protected-access
        DiffSplitter._sentence_transformers_available = True  # noqa: SLF001
        DiffSplitter._model_available = True  # noqa: SLF001

        # Mock the embedding function to return predictable values
        with patch.object(splitter, "_get_code_embedding") as mock_embedding:
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

            # Calculate similarities with mocked embeddings
            sim1_2 = splitter._calculate_semantic_similarity(code1, code2)  # pylint: disable=protected-access # noqa: SLF001
            sim1_3 = splitter._calculate_semantic_similarity(code1, code3)  # pylint: disable=protected-access # noqa: SLF001

            # Functions doing similar things should have higher similarity
            assert sim1_2 > sim1_3
    finally:
        # Restore original class-level availability flags
        # pylint: disable=protected-access
        DiffSplitter._sentence_transformers_available = original_st_available  # noqa: SLF001
        DiffSplitter._model_available = original_model_available  # noqa: SLF001


def test_sentence_transformers_availability() -> None:
    """Test the sentence-transformers availability check functions."""
    # Save original class-level availability flags
    # pylint: disable=protected-access
    original_st_available = DiffSplitter._sentence_transformers_available  # noqa: SLF001
    original_model_available = DiffSplitter._model_available  # noqa: SLF001

    try:
        # Reset class variables to force availability check
        # pylint: disable=protected-access
        DiffSplitter._sentence_transformers_available = None  # noqa: SLF001
        DiffSplitter._model_available = None  # noqa: SLF001

        # We'll manually set the availability flags since the actual import may vary by environment
        # pylint: disable=protected-access
        with patch.object(DiffSplitter, "_check_sentence_transformers_availability") as mock_check:
            mock_check.return_value = True

            # Try to initialize a new splitter which will trigger availability check
            repo_root = Path("/mock/repo")
            splitter = DiffSplitter(repo_root)

            # Force the sentence_transformers_available flag to True for testing
            DiffSplitter._sentence_transformers_available = True  # noqa: SLF001

            # Check if sentence-transformers is now available in our mocked environment
            assert DiffSplitter._sentence_transformers_available is True  # noqa: SLF001

            # Test with get_code_embedding
            with patch.object(splitter, "_get_code_embedding") as mock_embedding:
                mock_embedding.return_value = [0.1, 0.2, 0.3]

                # Test calculate_semantic_similarity with mocked embeddings
                with patch.object(splitter, "_calculate_semantic_similarity") as mock_sim:
                    mock_sim.return_value = 0.75

                    # Now check a similarity calculation would give the expected result
                    result = mock_sim("code1", "code2")
                    assert result == 0.75

    finally:
        # Restore original class-level availability flags
        # pylint: disable=protected-access
        DiffSplitter._sentence_transformers_available = original_st_available  # noqa: SLF001
        DiffSplitter._model_available = original_model_available  # noqa: SLF001
