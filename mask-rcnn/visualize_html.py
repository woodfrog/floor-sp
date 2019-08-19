import os.path as osp
from simple_html import HTML


def writeHTML(out_path, image_indices, image_filenames, base_dir):
    h = HTML('html')
    h.p('Results')
    h.br()
    t = h.table(border='1')
    for image_i in image_indices:
        r = t.tr()
        for image_filename in image_filenames:
            filename = str(image_i) + '_' + image_filename + '.png'
            img_path = osp.join(base_dir, filename)
            r.td().img(src=img_path, width='256')
    h.br()

    html_file = open(out_path, 'w')
    html_file.write(str(h))
    html_file.close()


def write_qualitative_table(out_path, image_indices, image_filenames, base_dir):
    h = HTML('html')
    h.p('Qualitative Results')
    h.br()
    t = h.table(border='1')
    r = t.tr()
    r.td('Inputs', align='center')
    r.td('Ground Truth', align='center')
    r.td('FloorNet', align='center')
    r.td('Ours(w/o E_data and E_consis)', align='center')
    r.td('Ours(w/o E_data)', align='center')
    r.td('Ours(1-round)', align='center')
    r.td('Ours(2-round)', align='center')

    for image_i in image_indices:
        r = t.tr()
        for image_filename in image_filenames:
            filename = str(image_i) + '_' + image_filename + '.png'
            img_path = osp.join(base_dir, filename)
            r.td().img(src=img_path, width='200')
    h.br()

    html_file = open(out_path, 'w')
    html_file.write(str(h))
    html_file.close()


if __name__ == '__main__':
    image_filenames = ['hsv', 'real_gt', 'floornet', 'floorplan_heuristics', 'floorplan_no_inter', 'floorplan',
                       'floorplan_round_2']
    image_indices = [i for i in range(94) if i not in [2, 5, 19, 32, 34, 38, 41]]

    write_qualitative_table(out_path='./full_qualitative.html', image_indices=image_indices,
                            image_filenames=image_filenames,
                            base_dir='./test_viz')
