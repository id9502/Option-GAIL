This is the initial version of the code.

1. For viewing the learning curve, please pip3 install tensorboard and then:

`cd <root_dir>`
`tensorboard --logdir ./result --port 8091 --bind_all`

after this, your result can be viewed by visiting http://127.0.0.1:8091 with any internet browser.

2. For graphically seeing what your model has learned, use <root_dir>/helper/render_learned_model.py by:

`cd <root_dir>`
`python3 -m helper.render_learned_model <full path to your exp>/xxx.torch <full path to your exp>/config.log`

after this you will see a gym rendered window playing your learned model on exp.
