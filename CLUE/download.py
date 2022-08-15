# 下载CLUE的数据集


CLUE = {
    'afqmc':'https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip',
    'tnews':'https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip',
    'iflytek':'https://storage.googleapis.com/cluebenchmark/tasks/iflytek_public.zip',
    'ocnli':'https://storage.googleapis.com/cluebenchmark/tasks/ocnli_public.zip',
    'cmnli':'https://storage.googleapis.com/cluebenchmark/tasks/cmnli_public.zip',
    'cluewsc2020':'https://storage.googleapis.com/cluebenchmark/tasks/cluewsc2020_public.zip',
    'csl':'https://storage.googleapis.com/cluebenchmark/tasks/csl_public.zip',
    'cmrc2018':'https://storage.googleapis.com/cluebenchmark/tasks/cmrc2018_public.zip',
    'chid':'https://storage.googleapis.com/cluebenchmark/tasks/chid_public.zip',
    'c3':'https://storage.googleapis.com/cluebenchmark/tasks/c3_public.zip',
    'clue_diagnostics':'https://storage.googleapis.com/cluebenchmark/tasks/clue_diagnostics_public.zip',
    
}








for filename, path in paths.items():
        print('reading file: {}'.format(filename))
        with open(path, 'r') as f:
            lines = f.readlines()
            url_list = []
            for line in lines:
                url_list.append(line.strip('\n'))
            print(url_list)
