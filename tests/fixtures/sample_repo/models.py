"""Sample models for testing."""

from __future__ import annotations

from datetime import UTC, datetime


class BaseModel:
	"""Base model with common fields."""

	def __init__(self) -> None:
		"""Initialize base model with timestamp fields."""
		self.created_at: datetime = datetime.now(tz=UTC)
		self.updated_at: datetime | None = None


class User(BaseModel):
	"""User model representing system users."""

	def __init__(self, name: str, email: str) -> None:
		"""
		Initialize user with name and email.

		Args:
		    name: User's full name
		    email: User's email address

		"""
		super().__init__()
		self.name: str = name
		self.email: str = email
		self.orders: list[Order] = []


class Order(BaseModel):
	"""Order model representing user purchases."""

	def __init__(self, user: User, total: float) -> None:
		"""
		Initialize order with user and total.

		Args:
		    user: User who placed the order
		    total: Total amount of the order

		"""
		super().__init__()
		self.order_id: str = ""
		self.user: User = user
		self.total: float = total
		self.items: list[OrderItem] = []


class OrderItem(BaseModel):
	"""Order item model representing items in an order."""

	def __init__(self, order: Order, product: Product, quantity: int) -> None:
		"""
		Initialize order item with order, product and quantity.

		Args:
		    order: Parent order
		    product: Product being ordered
		    quantity: Number of items ordered

		"""
		super().__init__()
		self.order: Order = order
		self.product: Product = product
		self.quantity: int = quantity
		self.price: float = 0.0


class Product(BaseModel):
	"""Product model representing items available for purchase."""

	def __init__(self, name: str, price: float) -> None:
		"""
		Initialize product with name and price.

		Args:
		    name: Product name
		    price: Product price

		"""
		super().__init__()
		self.name: str = name
		self.price: float = price
		self.description: str | None = None
