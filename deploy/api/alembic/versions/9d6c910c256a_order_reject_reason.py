"""order reject reason

Revision ID: 9d6c910c256a
Revises: 0f2acf6687cd
Create Date: 2026-11-10 02:25:15.559525

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9d6c910c256a'
down_revision: Union[str, Sequence[str], None] = '0f2acf6687cd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('orders', sa.Column('reject_reason', sa.String(length=200), nullable=True))
    op.execute("UPDATE orders SET reject_reason = 'unknown' WHERE status = 'rejected'")
    op.create_check_constraint(op.f('ck_orders_rejected_has_reason'), 'orders',
                               "status <> 'rejected' OR reject_reason IS NOT NULL")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(op.f('ck_orders_rejected_has_reason'), 'orders', type_='check')
    op.drop_column('orders', 'reject_reason')
