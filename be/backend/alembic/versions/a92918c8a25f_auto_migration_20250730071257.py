"""auto migration 20250730071257

Revision ID: a92918c8a25f
Revises: 944675144dfd
Create Date: 2025-07-30 07:12:58.495444

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a92918c8a25f'
down_revision: Union[str, Sequence[str], None] = '944675144dfd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('recommendations',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('recommended_items', sa.Text(), nullable=False),
    sa.Column('model_type', sa.Enum('COLLABORATIVE', 'CONTENT_BASED', 'DEEP_LEARNING', name='model_type_enum'), nullable=False),
    sa.Column('generated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('recommendations')
    # ### end Alembic commands ###
