import random

import pytest

from r2r import (
    IndexArgsHNSW,
    IndexArgsIVFFlat,
    IndexMeasure,
    IndexMethod,
    Vector,
    VectorEntry,
)
from r2r.base import DatabaseConfig, VectorSearchResult
from r2r.providers.database.postgres import PostgresVectorDBProvider


def generate_sample_entries(num_entries=100, dimension=128):
    categories = ["electronics", "books", "clothing", "home", "sports"]
    sample_entries = []
    for i in range(num_entries):
        vector = [random.uniform(-1, 1) for _ in range(dimension)]
        metadata = {
            "key": f"value_id_{i}",
            "category": random.choice(categories),
            "price": random.uniform(10, 1000),
            "numeric_field": random.randint(1, 100),
            "title": f"Product {i}",
            "description": f"Description for product {i}",
        }
        sample_entries.append(
            VectorEntry(id=f"id_{i}", vector=Vector(vector), metadata=metadata)
        )
    return sample_entries


@pytest.fixture
def vector_db_provider():
    config = DatabaseConfig(
        host="localhost",
        port=5432,
        user="test_user",
        password="test_password",
        db_name="test_db",
        extra_fields={"vecs_collection": "test_collection"},
    )
    return PostgresVectorDBProvider(config, dimension=128)


def test_search(vector_db_provider):
    # Mock data
    query_vector = [0.1] * 128
    filters = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"category": "electronics"}},
                    {"range": {"price": {"gte": 100, "lte": 500}}},
                ]
            }
        }
    }

    results = vector_db_provider.search(
        query_vector=query_vector,
        filters=filters,
        limit=5,
        _source=["title", "description", "price"],
        sort=[{"price": "desc"}],
    )

    assert isinstance(results, list)
    assert len(results) <= 5
    for result in results:
        assert isinstance(result, VectorSearchResult)
        assert "title" in result.metadata
        assert "description" in result.metadata
        assert "price" in result.metadata
        assert result.metadata["category"] == "electronics"
        assert 100 <= result.metadata["price"] <= 500


def test_hybrid_search(vector_db_provider):
    # Mock data
    query_text = "smartphone"
    query_vector = [0.1] * 128
    filters = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"category": "electronics"}},
                    {"range": {"price": {"gte": 100, "lte": 1000}}},
                ]
            }
        }
    }

    results = vector_db_provider.hybrid_search(
        query_text=query_text,
        query_vector=query_vector,
        filters=filters,
        limit=10,
        _source=["title", "description", "price"],
        sort=[{"price": "asc"}],
        full_text_weight=0.7,
        semantic_weight=0.3,
    )

    assert isinstance(results, list)
    assert len(results) <= 10
    for result in results:
        assert isinstance(result, VectorSearchResult)
        assert "title" in result.metadata
        assert "description" in result.metadata
        assert "price" in result.metadata
        assert result.metadata["category"] == "electronics"
        assert 100 <= result.metadata["price"] <= 1000

    # Check if results are sorted by price in ascending order
    prices = [result.metadata["price"] for result in results]
    assert prices == sorted(prices)


def test_search_with_invalid_filters(vector_db_provider):
    query_vector = [0.1] * 128
    invalid_filters = {"invalid": "filter"}

    with pytest.raises(
        Exception
    ):  # Replace with the specific exception you expect
        vector_db_provider.search(
            query_vector=query_vector, filters=invalid_filters
        )


def test_hybrid_search_with_invalid_source(vector_db_provider):
    query_text = "test"
    query_vector = [0.1] * 128

    with pytest.raises(
        Exception
    ):  # Replace with the specific exception you expect
        vector_db_provider.hybrid_search(
            query_text=query_text,
            query_vector=query_vector,
            _source=False,  # This should raise an exception as it's not supported
        )


@pytest.fixture
def vector_db_provider_with_data(vector_db_provider):
    for entry in sample_entries:
        vector_db_provider.vector.upsert(entry)
    return vector_db_provider


def test_search(vector_db_provider_with_data):
    query_vector = sample_entries[0].vector.data
    filters = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"category": "electronics"}},
                    {"range": {"price": {"gte": 100, "lte": 500}}},
                ]
            }
        }
    }

    results = vector_db_provider_with_data.search(
        query_vector=query_vector,
        filters=filters,
        limit=5,
        _source=["title", "description", "price", "category"],
        sort=[{"price": "desc"}],
    )

    assert isinstance(results, list)
    assert len(results) <= 5
    for result in results:
        assert isinstance(result, VectorSearchResult)
        assert "title" in result.metadata
        assert "description" in result.metadata
        assert "price" in result.metadata
        assert "category" in result.metadata
        assert result.metadata["category"] == "electronics"
        assert 100 <= result.metadata["price"] <= 500


def test_hybrid_search(vector_db_provider_with_data):
    query_text = "smartphone"
    query_vector = sample_entries[0].vector.data
    filters = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"category": "electronics"}},
                    {"range": {"price": {"gte": 100, "lte": 1000}}},
                ]
            }
        }
    }

    results = vector_db_provider_with_data.hybrid_search(
        query_text=query_text,
        query_vector=query_vector,
        filters=filters,
        limit=10,
        _source=["title", "description", "price", "category"],
        sort=[{"price": "asc"}],
        full_text_weight=0.7,
        semantic_weight=0.3,
    )

    assert isinstance(results, list)
    assert len(results) <= 10
    for result in results:
        assert isinstance(result, VectorSearchResult)
        assert "title" in result.metadata
        assert "description" in result.metadata
        assert "price" in result.metadata
        assert "category" in result.metadata
        assert result.metadata["category"] == "electronics"
        assert 100 <= result.metadata["price"] <= 1000

    # Check if results are sorted by price in ascending order
    prices = [result.metadata["price"] for result in results]
    assert prices == sorted(prices)


@pytest.mark.parametrize("db_fixture", ["pg_vector_db"])
def test_search_with_complex_filters(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    for entry in sample_entries:
        db.vector.upsert(entry)

    complex_filters = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"key": f"value_id_{0}"}},
                    {"range": {"numeric_field": {"gte": 5, "lte": 10}}},
                ],
                "should": [
                    {"term": {"category": "electronics"}},
                    {"term": {"category": "books"}},
                ],
                "must_not": [{"term": {"status": "out_of_stock"}}],
            }
        }
    }

    query_vector = sample_entries[0].vector.data
    results = db.vector.search(
        query_vector=query_vector, filters=complex_filters
    )
    assert len(results) > 0
    for result in results:
        assert result.metadata["key"] == f"value_id_{0}"
        assert 5 <= result.metadata.get("numeric_field", 0) <= 10
        assert result.metadata.get("category") in ["electronics", "books"]
        assert result.metadata.get("status") != "out_of_stock"


@pytest.mark.parametrize("db_fixture", ["pg_vector_db"])
def test_search_with_source_filtering(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    for entry in sample_entries:
        db.vector.upsert(entry)

    query_vector = sample_entries[0].vector.data
    results = db.vector.search(query_vector=query_vector, _source=["key"])
    assert len(results) > 0
    for result in results:
        assert "key" in result.metadata
        assert len(result.metadata) == 1


@pytest.mark.parametrize("db_fixture", ["pg_vector_db"])
def test_search_with_custom_sorting(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    for i, entry in enumerate(sample_entries):
        entry.metadata["numeric_field"] = i
        db.vector.upsert(entry)

    query_vector = sample_entries[0].vector.data
    results = db.vector.search(
        query_vector=query_vector, sort=[{"numeric_field": "desc"}]
    )
    assert len(results) > 0
    for i in range(1, len(results)):
        assert (
            results[i - 1].metadata["numeric_field"]
            >= results[i].metadata["numeric_field"]
        )


@pytest.mark.parametrize("db_fixture", ["pg_vector_db"])
def test_search_with_different_measures(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    for entry in sample_entries:
        db.vector.upsert(entry)

    query_vector = sample_entries[0].vector.data
    measures = [
        IndexMeasure.cosine_distance,
        IndexMeasure.l2_distance,
        IndexMeasure.max_inner_product,
    ]

    for measure in measures:
        results = db.vector.search(query_vector=query_vector, measure=measure)
        assert len(results) > 0


@pytest.mark.parametrize("db_fixture", ["pg_vector_db"])
def test_create_and_search_with_different_indexes(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    for entry in sample_entries:
        db.vector.upsert(entry)

    query_vector = sample_entries[0].vector.data

    # Test IVFFlat index
    db.vector.create_index(
        IndexMethod.ivfflat,
        IndexMeasure.cosine_distance,
        IndexArgsIVFFlat(n_lists=10),
    )
    results_ivf = db.vector.search(query_vector=query_vector)
    assert len(results_ivf) > 0

    # Test HNSW index (if supported)
    if db.vector.client._supports_hnsw():
        db.vector.create_index(
            IndexMethod.hnsw,
            IndexMeasure.cosine_distance,
            IndexArgsHNSW(m=16, ef_construction=64),
            replace=True,
        )
        results_hnsw = db.vector.search(query_vector=query_vector)
        assert len(results_hnsw) > 0


@pytest.mark.parametrize("db_fixture", ["pg_vector_db"])
def test_search_with_probes_and_ef_search(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    for entry in sample_entries:
        db.vector.upsert(entry)

    query_vector = sample_entries[0].vector.data

    # Test with IVFFlat index and custom probes
    db.vector.create_index(
        IndexMethod.ivfflat,
        IndexMeasure.cosine_distance,
        IndexArgsIVFFlat(n_lists=10),
    )
    results_ivf = db.vector.search(query_vector=query_vector, probes=5)
    assert len(results_ivf) > 0

    # Test with HNSW index and custom ef_search (if supported)
    if db.vector.client._supports_hnsw():
        db.vector.create_index(
            IndexMethod.hnsw,
            IndexMeasure.cosine_distance,
            IndexArgsHNSW(m=16, ef_construction=64),
            replace=True,
        )
        results_hnsw = db.vector.search(
            query_vector=query_vector, ef_search=50
        )
        assert len(results_hnsw) > 0


@pytest.mark.parametrize("db_fixture", ["pg_vector_db"])
def test_hybrid_search(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    for entry in sample_entries:
        db.vector.upsert(entry)

    query_text = "sample query"
    query_vector = sample_entries[0].vector.data
    results = db.vector.hybrid_search(
        query_text=query_text,
        query_vector=query_vector,
        limit=5,
        filters={"key": f"value_id_{0}"},
        _source=["key"],
        sort=[{"numeric_field": "asc"}],
        full_text_weight=0.7,
        semantic_weight=0.3,
    )

    assert len(results) > 0
    assert all("key" in result.metadata for result in results)
    assert all(result.metadata["key"] == f"value_id_{0}" for result in results)
    for i in range(1, len(results)):
        assert results[i - 1].metadata.get("numeric_field", 0) <= results[
            i
        ].metadata.get("numeric_field", 0)
