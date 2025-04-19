"""Tests for enhanced semantic chunking functionality in DiffSplitter."""

from pathlib import Path

import pytest

from codemap.git.commit.diff_splitter import DiffSplitter
from codemap.git.utils.git_utils import GitDiff


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

    # Verify the chunking
    assert len(chunks) > 1  # Should be split into multiple chunks

    # Check that imports, functions, classes and main block are in separate chunks
    import_chunk = next((c for c in chunks if "import" in c), None)
    function_chunk = next((c for c in chunks if "function1" in c), None)
    class_chunk = next((c for c in chunks if "class Example" in c), None)
    main_chunk = next((c for c in chunks if "__main__" in c), None)

    assert import_chunk is not None
    assert function_chunk is not None
    assert class_chunk is not None
    assert main_chunk is not None


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

    # Verify the chunking
    assert len(chunks) > 1  # Should be split into multiple chunks

    # Check that imports, functions, and classes are in separate chunks
    import_chunk = next((c for c in chunks if "import React" in c), None)
    function_chunk = next((c for c in chunks if "function calculateTotal" in c), None)
    class_chunk = next((c for c in chunks if "class ShoppingCart" in c), None)
    export_chunk = next((c for c in chunks if "export default" in c), None)

    assert import_chunk is not None
    assert function_chunk is not None
    assert class_chunk is not None
    assert export_chunk is not None


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

    # Verify the chunking
    assert len(chunks) > 2  # Should have multiple chunks across both files

    # Verify that Python and JavaScript files were properly split
    py_chunks = [c for c in chunks if c.files[0] == "example.py"]
    js_chunks = [c for c in chunks if c.files[0] == "utils.js"]

    assert len(py_chunks) >= 2  # Python file should have multiple chunks
    assert len(js_chunks) >= 2  # JavaScript file should have multiple chunks


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
        import sentence_transformers  # noqa: F401
    except ImportError:
        pytest.skip("sentence-transformers not installed")

    # Initialize diff splitter with a mock repo root
    repo_root = Path("/mock/repo")
    splitter = DiffSplitter(repo_root)

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

    # Calculate similarities
    sim1_2 = splitter._calculate_semantic_similarity(code1, code2)  # pylint: disable=protected-access # noqa: SLF001
    sim1_3 = splitter._calculate_semantic_similarity(code1, code3)  # pylint: disable=protected-access # noqa: SLF001

    # Functions doing similar things should have higher similarity
    assert sim1_2 > sim1_3
