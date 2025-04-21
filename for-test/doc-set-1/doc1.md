---
title: Clustered Indexes
summary: Learn the concept, user scenarios, usages, limitations, and compatibility of clustered indexes.
---

# Clustered Indexes

TiDB supports the clustered index feature since v5.0. This feature controls how data is stored in tables containing primary keys. It provides TiDB the ability to organize tables in a way that can improve the performance of certain queries.

The term _clustered_ in this context refers to the _organization of how data is stored_ and not _a group of database servers working together_. Some database management systems refer to clustered index tables as _index-organized tables_ (IOT).

Currently, tables containing primary keys in TiDB are divided into the following two categories:

- `NONCLUSTERED`: The primary key of the table is non-clustered index. In tables with non-clustered indexes, the keys for row data consist of internal `_tidb_rowid` implicitly assigned by TiDB. Because primary keys are essentially unique indexes, tables with non-clustered indexes need at least two key-value pairs to store a row, which are:
    - `_tidb_rowid` (key) - row data (value)
    - Primary key data (key) - `_tidb_rowid` (value)
- `CLUSTERED`: The primary key of the table is clustered index. In tables with clustered indexes, the keys for row data consist of primary key data given by the user. Therefore, tables with clustered indexes need only one key-value pair to store a row, which is:
    - Primary key data (key) - row data (value)

> **Note:**
>
> TiDB supports clustering only by a table's `PRIMARY KEY`. With clustered indexes enabled, the terms _the_ `PRIMARY KEY` and _the clustered index_ might be used interchangeably. `PRIMARY KEY` refers to the constraint (a logical property), and clustered index describes the physical implementation of how the data is stored.