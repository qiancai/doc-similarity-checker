---
title: Clustered Indexes
summary: Introduction to Clustered Indexes.
---

# Clustered Indexes

TiDB supports the clustered index feature starting from v5.0, which controls how data is stored in tables containing primary keys. This feature provides TiDB the ability to organize tables in a way that can improve the performance of certain queries.

The term _clustered_ in this context refers to the _organization of how data is stored_ and not _a group of database servers working together_. Some database management systems refer to clustered index tables as _index-organized tables_ (IOT).

Currently, there are two categories for tables containing primary keys in TiDB:

- `NONCLUSTERED`: The primary key of the table is non-clustered index. In tables with non-clustered indexes, the keys for row data consist of internal `_tidb_rowid` implicitly assigned by TiDB. Because primary keys are essentially unique indexes, tables with non-clustered indexes need at least two key-value pairs to store a row, which are:
    - `_tidb_rowid` (key) - row data (value)
    - Primary key data (key) - `_tidb_rowid` (value)
- `CLUSTERED`: The primary key of the table is clustered index. In tables with clustered indexes, the keys for row data consist of primary key data given by the user. Therefore, tables with clustered indexes need only one key-value pair to store a row, which is:
    - Primary key data (key) - row data (value)

> **Note:**
>
> This is an example note.