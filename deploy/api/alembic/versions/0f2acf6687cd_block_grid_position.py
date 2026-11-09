"""block grid position

Adds yard_block.grid_row / grid_col. The arrangement comes from the deployed model, so
existing layout rows cannot be backfilled: they are deleted (only when no container has ever
been placed) and must be recreated with `python -m app.db.seed`.

Revision ID: 0f2acf6687cd
Revises: 876fe4a3a711
Create Date: 2026-11-10 00:07:01.633081

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0f2acf6687cd'
down_revision: Union[str, Sequence[str], None] = '876fe4a3a711'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()
    if conn.scalar(sa.text("SELECT count(*) FROM yard_block")):
        if conn.scalar(sa.text("SELECT count(*) FROM container_location")):
            raise RuntimeError("yard_block has containers placed in it; cannot clear the layout")
        op.execute("DELETE FROM yard_state")
        op.execute("DELETE FROM slot")
        op.execute("DELETE FROM yard_block")
        print("Yard layout cleared: run `python -m app.db.seed` to recreate it.")

    op.add_column('yard_block', sa.Column('grid_row', sa.SmallInteger(), nullable=False))
    op.add_column('yard_block', sa.Column('grid_col', sa.SmallInteger(), nullable=False))
    op.create_unique_constraint('uq_block_grid_position', 'yard_block', ['grid_row', 'grid_col'])
    op.create_check_constraint(op.f('ck_yard_block_grid_nonneg'), 'yard_block',
                               'grid_row >= 0 AND grid_col >= 0')


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(op.f('ck_yard_block_grid_nonneg'), 'yard_block', type_='check')
    op.drop_constraint('uq_block_grid_position', 'yard_block', type_='unique')
    op.drop_column('yard_block', 'grid_col')
    op.drop_column('yard_block', 'grid_row')
