# Mypy Guide

## Common Mypy Errors and Fixes

### 1. no-any-return

**Problem:** Function returns `Any` type when a specific type is expected.

```python
# Error: Returning Any from function declared to return "str"  [no-any-return]
def get_name(data: dict) -> str:
    return data.get("name")  # dict.get() returns Any
```

**Fix:** Use `cast()` or explicit conversion like `str()`:

```python
from typing import cast

# Option 1: Use cast() when you know the type
def get_name(data: dict) -> str:
    return cast(str, data.get("name"))

# Option 2: Use str() for string conversion
def get_name(data: dict) -> str:
    return str(data.get("name", ""))

# Option 3: Add proper type narrowing
def get_name(data: dict) -> str:
    name = data.get("name")
    if isinstance(name, str):
        return name
    return ""
```

---

### 2. import-not-found / import-untyped

**Problem:** Module has no type stubs or is not typed.

```python
# Error: Cannot find implementation or library stub for module named "redis"  [import-not-found]
# Error: Library stubs not installed for "redis"  [import-untyped]
import redis
```

**Fix:** Install type stubs or configure mypy to ignore:

```bash
# Option 1: Install types-* packages
pip install types-redis types-requests types-PyYAML

# Common type stub packages:
pip install types-redis
pip install types-requests
pip install types-PyYAML
pip install types-python-dateutil
pip install types-passlib
```

```ini
# Option 2: In mypy.ini or pyproject.toml, ignore missing imports
[mypy]
ignore_missing_imports = True

# Or per-module:
[mypy-redis.*]
ignore_missing_imports = True
```

```python
# Option 3: Inline ignore for specific imports
import redis  # type: ignore[import-untyped]
```

---

### 3. arg-type with SQLAlchemy Column Comparisons

**Problem:** SQLAlchemy Column comparisons return `ColumnElement[bool]` not `bool`.

```python
# Error: Argument 1 to "filter" has incompatible type "ColumnElement[bool]"; expected "bool"  [arg-type]
from sqlalchemy import Column, String
from sqlalchemy.orm import Session

class User(Base):
    id = Column(String, primary_key=True)
    name = Column(String)

def get_user(db: Session, name: str) -> User | None:
    return db.query(User).filter(User.name == name).first()  # Error here
```

**Fix:** Use `# type: ignore[arg-type]` comment:

```python
def get_user(db: Session, name: str) -> User | None:
    return db.query(User).filter(User.name == name).first()  # type: ignore[arg-type]

# For multiple conditions:
def search_users(db: Session, name: str, active: bool) -> list[User]:
    return db.query(User).filter(
        User.name == name,  # type: ignore[arg-type]
        User.is_active == active,  # type: ignore[arg-type]
    ).all()

# Alternative: ignore at statement level
def get_user(db: Session, name: str) -> User | None:
    query = db.query(User).filter(User.name == name)  # type: ignore[arg-type]
    return query.first()
```

---

### 4. E402 Module Imports Not at Top

**Problem:** Imports not at top of file (common in Alembic migrations and conditional imports).

```python
# Error: E402 module level import not at top of file
import sys
sys.path.insert(0, "/some/path")

import my_module  # E402 error here
```

**Fix:** Use `# noqa: E402` comment:

```python
# alembic/env.py style file
import sys
from pathlib import Path

# Add parent directory to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import Base  # noqa: E402
from app.config import settings  # noqa: E402
```

```python
# Conditional imports example
import platform

if platform.system() == "Windows":
    import winreg  # noqa: E402
else:
    import posix  # noqa: E402
```

```python
# Full alembic/env.py example
from logging.config import fileConfig
import sys
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from alembic import context

# Add app to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import Base  # noqa: E402
from app.models import *  # noqa: E402, F401
from app.config import get_settings  # noqa: E402
```

---

## Quick Reference

| Error Code | Fix |
|------------|-----|
| `no-any-return` | `cast(Type, value)` or `str(value)` |
| `import-not-found` | `pip install types-*` or `ignore_missing_imports = True` |
| `import-untyped` | `# type: ignore[import-untyped]` |
| `arg-type` (SQLAlchemy) | `# type: ignore[arg-type]` |
| `E402` | `# noqa: E402` |

---

## SQLAlchemy and Mypy

SQLAlchemy's ORM doesn't play nicely with mypy out of the box. Here are the practical patterns.

### Column Type Annotations

SQLAlchemy columns return `Column[T]` but you want to annotate them as `T`. Use `# type: ignore[assignment]`:

```python
from sqlalchemy import Column, String, Integer, DateTime
from datetime import datetime

class MyModel(Base):
    __tablename__ = "my_table"

    id: int = Column(Integer, primary_key=True)  # type: ignore[assignment]
    name: str = Column(String(255), nullable=False)  # type: ignore[assignment]
    created_at: datetime = Column(DateTime, default=datetime.utcnow)  # type: ignore[assignment]
```

### Enum Columns

Same pattern for enum columns:

```python
from sqlalchemy import Column, Enum
from enum import Enum as PyEnum

class ArtifactKind(PyEnum):
    FILE = "file"
    DIRECTORY = "directory"

class Artifact(Base):
    __tablename__ = "artifacts"

    kind: ArtifactKind = Column(Enum(ArtifactKind), nullable=False)  # type: ignore[assignment]
```

### Column Comparisons (bool vs ColumnElement[bool])

When comparing columns in queries, SQLAlchemy returns `ColumnElement[bool]`, not `bool`. This matters in filter conditions:

```python
from sqlalchemy.sql.elements import ColumnElement

# In a query filter - this returns ColumnElement[bool], not bool
query.filter(MyModel.name == "test")  # OK - SQLAlchemy handles this

# But if you're building conditions dynamically and mypy complains:
def build_filter(name: str) -> ColumnElement[bool]:
    return MyModel.name == name  # type: ignore[return-value]

# Or just ignore inline:
condition: bool = MyModel.name == name  # type: ignore[assignment]
```

### Converting Column[str] to str

When passing column values to functions that expect `str`, use `str()`:

```python
# Function expects str
def process_name(name: str) -> str:
    return name.upper()

# Column value in a loop/query result
for artifact in session.query(Artifact).all():
    # artifact.name is technically Column[str] to mypy
    result = process_name(str(artifact.name))

    # Or if you're sure it's already resolved (post-query):
    result = process_name(artifact.name)  # type: ignore[arg-type]
```

### Nullable Columns

For nullable columns, use `Optional`:

```python
from typing import Optional

class MyModel(Base):
    description: Optional[str] = Column(String(500), nullable=True)  # type: ignore[assignment]
```

### SQLAlchemy Quick Reference

| Pattern | Example |
|---------|---------|
| Basic column | `name: str = Column(String)  # type: ignore[assignment]` |
| Enum column | `kind: MyEnum = Column(Enum(MyEnum))  # type: ignore[assignment]` |
| Nullable | `desc: Optional[str] = Column(String, nullable=True)  # type: ignore[assignment]` |
| Comparison return | `def f() -> ColumnElement[bool]: return Model.x == y  # type: ignore[return-value]` |
| Column to str | `str(model.column)` or `model.column  # type: ignore[arg-type]` |
