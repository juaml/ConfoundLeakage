import sys
import re


def get_cell_close_open(cell_title, cell_type, cell_tags):

    if cell_type == "code":
        cell_open = '```{code-cell}\n'

        if cell_tags is not None:
            cell_tags = cell_tags.replace('"', '').replace("'", '')
            cell_open += f':tags: {cell_tags}\n'
        if not cell_title.isspace():
            cell_open += f':title: {cell_title}\n'

        cell_close = '```\n\n'
        remove_comment = False

    elif cell_type == "markdown":
        cell_open = '+++'

        if cell_tags is not None and not cell_title.isspace():
            cell_open += (
                ' {'
                + f'"tags": {cell_tags}, '
                + f'"title": "{cell_title}"'
                + '}'
            )

        elif cell_tags is not None:
            cell_open += (
                ' {'
                + f'"tags": {cell_tags}'
                + '}'
            )
        elif not cell_title.isspace():
            cell_open += (
                ' {'
                + f'"title": "{cell_title}"'
                + '}'
            )
        cell_open += '\n'

        cell_close = '\n'
        remove_comment = True

    elif cell_type == "tabbed":
        cell_open = (
            '````{tabbed} '
            f'{cell_title}\n'
        )
        cell_close = '````\n\n'

        remove_comment = True
    elif cell_type == "tabbed-code":
        cell_open = (
            '````{tabbed} '
            f'{cell_title}\n'
            '`{ '
        )
        cell_close = '`}\n````\n\n'

        remove_comment = False
    else:
        raise ValueError(f'cell_type {cell_type} is not a valid type')
    return cell_open, cell_close, remove_comment


def convert(filename):

    pattern_cell_start = re.compile(r'#\s*%%')
    pattern_comment = re.compile(r'#\s*')
    with open(filename, 'r') as f:

        close_previous_cell = ''
        new_text = (
            '---\n'
            'jupytext:\n'
            '  text_representation:\n'
            '    extension: .md\n'
            '    format_name: myst\n'
            '    format_version: 0.13\n'
            '    jupytext_version: 1.11.5\n'
            'kernelspec:\n'
            '  display_name: Python 3\n'
            '  language: python\n'
            '  name: python3\n'
            '---\n'
        )
        remove_comment = False
        header = True
        for line in f:

            if header:
                if line.startswith('# ---'):
                    remove_comment = not remove_comment

                # new_text += line.replace('# ', '')

                if not remove_comment:
                    header = False

            elif pattern_cell_start.match(line):

                # get information
                content = pattern_cell_start.split(line)[1]
                title_and_type, *tags = content.split('tags=')
                cell_tags = None if tags == [] else tags[0][:-1]
                cell_title, *cell_type = title_and_type.split('[')
                cell_type = 'code' if cell_type == [] else cell_type[0][:-2]
                open_cell, close_cell, remove_comment = get_cell_close_open(
                    cell_title, cell_type, cell_tags)

                # add text to file
                new_text += close_previous_cell + open_cell

                # prepare next cell
                close_previous_cell = close_cell

            else:
                if remove_comment:
                    new_text += pattern_comment.sub('', line, 1)
                else:
                    new_text += line

        if line != close_cell:
            new_text += close_cell

    with open(filename.replace('.py', '.md'), 'w') as f:
        f.write(new_text)


if __name__ == '__main__':
    convert(sys.argv[1])
