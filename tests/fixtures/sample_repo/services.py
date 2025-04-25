"""Service layer for handling business logic."""

from __future__ import annotations

from .models import Order, OrderItem, Product, User


class OrderService:
	"""Service for handling order-related operations."""

	def create_order(self, user: User, items: list[tuple[Product, int]]) -> Order:
		"""Create a new order for a user with the given items."""
		total = sum(product.price * quantity for product, quantity in items)
		order = Order(user, total)

		for product, quantity in items:
			order_item = OrderItem(order, product, quantity)
			order_item.price = product.price * quantity
			order.items.append(order_item)

		user.orders.append(order)
		return order


class UserService:
	"""Service for handling user-related operations."""

	def get_user_orders(self, user: User) -> list[Order]:
		"""Get all orders for a user."""
		return user.orders

	def get_total_spent(self, user: User) -> float:
		"""Calculate total amount spent by user."""
		return sum(order.total for order in user.orders)


class ProductService:
	"""Service for handling product-related operations."""

	def get_products_by_price_range(
		self,
		min_price: float | None = None,
		max_price: float | None = None,
	) -> list[Product]:
		"""Get products within the specified price range."""
		products: list[Product] = []  # In real code, this would query a database
		return [
			p
			for p in products
			if (min_price is None or p.price >= min_price) and (max_price is None or p.price <= max_price)
		]
