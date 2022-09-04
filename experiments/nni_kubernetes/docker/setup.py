from setuptools import find_packages, setup


def main():
    import ml_benchmark as package

    setup(
        name='ml_benchmark',
        version=package.__version__,
        description='ml_benchmark - A ML-Job for Benchmarking.',
        license='MIT',
        author='Michael Gebauer, Sebastian Werner',
        author_email='gebauerm23@gmail.com',
        url=package.URL,
        download_url=f'{package.URL}/archive/{package.__version__}.tar.gz',
        packages=find_packages(),
        install_requires=package.install_requires,
        dependency_links=[""],
        python_requires=">=3.6",
        include_package_data=True,
        extras_require={"test": package.test_install_requires},
        long_description="/README.md"
        )


if __name__ == "__main__":
    main()
