"""auto migration 20250730063821

Revision ID: 03cb20dc4ce3
Revises: f48f554cbf20
Create Date: 2025-07-30 06:38:22.891372

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '03cb20dc4ce3'
down_revision: Union[str, Sequence[str], None] = 'f48f554cbf20'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    pass
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    pass
    # ### end Alembic commands ###
