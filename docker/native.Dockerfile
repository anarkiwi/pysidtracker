ARG PYTHON_VERSION=3.12

FROM rust:1.99-slim-trixie AS rust

FROM python:${PYTHON_VERSION}-slim-trixie AS pydeps
WORKDIR /src
COPY pyproject.toml .
# pysidtracker is installed from the wheel built below, so only its dependencies are.
RUN python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; \
print('\n'.join(p['dependencies'] + p['optional-dependencies']['dev']))" > /requirements.txt \
    && pip install --no-cache-dir -r /requirements.txt

FROM python:${PYTHON_VERSION}-slim-trixie AS toolchain
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo PATH=/usr/local/cargo/bin:$PATH
COPY --from=rust /usr/local/rustup /usr/local/rustup
COPY --from=rust /usr/local/cargo /usr/local/cargo
RUN pip install --no-cache-dir maturin

FROM toolchain AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates/specvm-core/Cargo.toml crates/specvm-core/
COPY crates/specvm-py/Cargo.toml crates/specvm-py/
RUN mkdir -p crates/specvm-core/src crates/specvm-py/src \
    && touch crates/specvm-core/src/lib.rs crates/specvm-py/src/lib.rs \
    && cargo fetch --locked
COPY pyproject.toml README.md LICENSE ./
COPY crates crates
COPY src src
RUN maturin build --release --locked -i python -o /wheels

FROM pydeps AS test
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-deps /wheels/*.whl \
    && python -c "from pysidtracker.playspec import native; assert native.available()"
# Only the tests and the examples they read are copied, so imports resolve to the installed wheel.
COPY pyproject.toml strain-baseline.json ./
COPY tests tests
COPY examples examples
CMD ["pytest", "tests", "-n", "auto", "-q", "--cov=pysidtracker", "--cov-fail-under=85"]
