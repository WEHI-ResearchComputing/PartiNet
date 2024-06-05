import click

@click.group()
def main():
    pass

@main.command()
def preprocess():
    click.echo("This will preprocess the micrographs.")

@main.command()
def train1():
    click.echo("This will perform DynamicDet training step 1.")

@main.command()
def train2():
    click.echo("This will perform DynamicDet training step 2.")

@main.command()
def infer():
    click.echo("This will perform DynamicDet inference.")