"""auto migration 20250730095432

Revision ID: e1891ba7d735
Revises: 87965495b2d3
Create Date: 2025-07-30 09:54:34.980601

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision: str = 'e1891ba7d735'
down_revision: Union[str, Sequence[str], None] = '87965495b2d3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('users', 'gender',
               existing_type=mysql.VARCHAR(length=10),
               type_=sa.Enum('남', '여', name='gender_enum', native_enum=False),
               existing_nullable=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('users', 'gender',
               existing_type=sa.Enum('남', '여', name='gender_enum', native_enum=False),
               type_=mysql.VARCHAR(length=10),
               existing_nullable=True)
    # ### end Alembic commands ###
