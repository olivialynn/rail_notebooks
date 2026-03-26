GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
```GPz.ipynb`` <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/estimation_examples/GPz.ipynb>`__
in the ``interactive_examples/estimation_examples/`` folder for a
version of this notebook in interactive mode.

A quick demo of running GPz on the typical test data. You should have
installed rail_gpz_v1 (we highly recommend that you do this from within
a custom conda environment so that all dependencies for package versions
are met), either by cloning and installing from github, or with:

::

   pip install pz-rail-gpz-v1

As RAIL is a namespace package, installing rail_gpz_v1 will make
``GPzInformer`` and ``GPzEstimator`` available, and they can be imported
via:

::

   from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

Let’s start with all of our necessary imports:

.. code:: ipython3

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import rail
    import tables_io
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = tables_io.read(trainFile)
    test_data = tables_io.read(testFile)

Now, we need to set up the stage that will run GPz. We begin by defining
a dictionary with the config options for the algorithm. There are
sensible defaults set, we will override several of these as an example
of how to do this. Config parameters not set in the dictionary will
automatically be set to their default values.

.. code:: ipython3

    gpz_train_dict = dict(n_basis=60, trainfrac=0.8, csl_method="normal", max_iter=150, hdf5_groupname="photometry") 

Let’s set up the training stage. We need to provide a name for the stage
for ceci, as well as a name for the model file that will be written by
the stage. We also include the arguments in the dictionary we wrote
above as additional arguments:

.. code:: ipython3

    # set up the stage to run our GPZ_training
    pz_train = GPzInformer.make_stage(name="GPz_Train", model="GPz_model.pkl", **gpz_train_dict)

We are now ready to run the stage to create the model. We will use the
training data from ``test_dc2_training_9816.hdf5``, which contains
10,225 galaxies drawn from healpix 9816 from the cosmoDC2_v1.1.4
dataset, to train the model. Note that we read this data in called
``train_data`` in the DataStore. Note that we set ``trainfrac`` to 0.8,
so 80% of the data will be used in the “main” training, but 20% will be
reserved by ``GPzInformer`` to determine a SIGMA parameter. We set
``max_iter`` to 150, so we will see 150 steps where the stage tries to
maximize the likelihood. We run the stage as follows:

.. code:: ipython3

    %%time
    pz_train.inform(training_data)


.. parsed-literal::

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4100014e-01	 3.1953600e-01	-3.3069283e-01	 3.2413977e-01	[-3.3881252e-01]	 4.2984009e-01


.. parsed-literal::

       2	-2.7277534e-01	 3.1016517e-01	-2.4988423e-01	 3.1446585e-01	[-2.6064681e-01]	 2.1564174e-01


.. parsed-literal::

       3	-2.2897873e-01	 2.8936209e-01	-1.8727540e-01	 2.9027187e-01	[-1.8884595e-01]	 2.7189541e-01
       4	-1.8027273e-01	 2.6492094e-01	-1.3648103e-01	 2.6257991e-01	[-1.2340785e-01]	 1.8966818e-01


.. parsed-literal::

       5	-9.7219372e-02	 2.5511356e-01	-6.7040779e-02	 2.5741596e-01	[-6.9294452e-02]	 2.0305777e-01


.. parsed-literal::

       6	-6.5709651e-02	 2.5037822e-01	-3.6906776e-02	 2.5260296e-01	[-4.3737789e-02]	 2.0758581e-01
       7	-4.7085202e-02	 2.4753370e-01	-2.3406929e-02	 2.5110043e-01	[-3.4971015e-02]	 2.0882916e-01


.. parsed-literal::

       8	-3.3278333e-02	 2.4516052e-01	-1.3369173e-02	 2.4971933e-01	[-2.8773989e-02]	 2.0389509e-01


.. parsed-literal::

       9	-2.2327975e-02	 2.4314623e-01	-4.4301123e-03	 2.4736622e-01	[-1.8590931e-02]	 2.1024156e-01
      10	-1.4302445e-02	 2.4155003e-01	 1.3053701e-03	 2.4504953e-01	[-1.1954201e-02]	 1.7322803e-01


.. parsed-literal::

      11	-8.7187955e-03	 2.4060203e-01	 6.6886662e-03	 2.4432883e-01	[-5.9690220e-03]	 1.9865322e-01
      12	-6.0198698e-03	 2.4011586e-01	 9.1535357e-03	 2.4404544e-01	[-4.6072501e-03]	 1.9875193e-01


.. parsed-literal::

      13	-2.9832494e-03	 2.3952156e-01	 1.2086348e-02	 2.4392335e-01	[-3.2230341e-03]	 1.6920996e-01
      14	 9.1628189e-03	 2.3714007e-01	 2.5084453e-02	 2.4337468e-01	[ 4.6199854e-03]	 1.9528723e-01


.. parsed-literal::

      15	 1.0545911e-01	 2.2563408e-01	 1.2961080e-01	 2.3043656e-01	[ 1.2002710e-01]	 3.1300926e-01
      16	 2.0629495e-01	 2.1647435e-01	 2.3267439e-01	 2.2133325e-01	[ 2.1481247e-01]	 1.9524336e-01


.. parsed-literal::

      17	 2.8871206e-01	 2.0960430e-01	 3.1868798e-01	 2.1516405e-01	[ 2.9018522e-01]	 3.1065607e-01


.. parsed-literal::

      18	 3.4290772e-01	 2.0750293e-01	 3.7708941e-01	 2.1063442e-01	[ 3.3819249e-01]	 2.0894623e-01
      19	 3.9318306e-01	 2.0637406e-01	 4.2818877e-01	 2.1296795e-01	[ 3.7693678e-01]	 1.8654966e-01


.. parsed-literal::

      20	 4.3964254e-01	 2.0607238e-01	 4.7504943e-01	 2.1375724e-01	[ 4.1769768e-01]	 1.7740178e-01


.. parsed-literal::

      21	 4.9614571e-01	 2.0204589e-01	 5.3240414e-01	 2.0950019e-01	[ 4.6674311e-01]	 2.0467496e-01


.. parsed-literal::

      22	 5.7489883e-01	 1.9939639e-01	 6.1458408e-01	 2.0494367e-01	[ 5.1546524e-01]	 2.0811224e-01
      23	 6.1482397e-01	 2.0075498e-01	 6.5576103e-01	 2.0515511e-01	[ 5.2685712e-01]	 1.9450331e-01


.. parsed-literal::

      24	 6.5094153e-01	 1.9519950e-01	 6.9029432e-01	 2.0006617e-01	[ 5.8142147e-01]	 1.9865060e-01


.. parsed-literal::

      25	 6.8907221e-01	 1.9205073e-01	 7.2789801e-01	 1.9669963e-01	[ 6.2447742e-01]	 2.0258856e-01
      26	 7.0324413e-01	 1.9849463e-01	 7.4069546e-01	 2.0389939e-01	[ 6.5284350e-01]	 1.9924116e-01


.. parsed-literal::

      27	 7.4548382e-01	 1.9246372e-01	 7.8370991e-01	 1.9779076e-01	[ 6.8544484e-01]	 2.0167851e-01
      28	 7.6405649e-01	 1.9117422e-01	 8.0305314e-01	 1.9608084e-01	[ 7.0111857e-01]	 1.9578552e-01


.. parsed-literal::

      29	 7.8689575e-01	 1.9106481e-01	 8.2681588e-01	 1.9681416e-01	[ 7.2322480e-01]	 1.9545007e-01
      30	 8.1272355e-01	 1.8786189e-01	 8.5363716e-01	 1.9451740e-01	[ 7.4801882e-01]	 1.9683385e-01


.. parsed-literal::

      31	 8.3141965e-01	 1.8912830e-01	 8.7164222e-01	 1.9607119e-01	[ 7.7507884e-01]	 2.0419574e-01
      32	 8.6212553e-01	 1.8578476e-01	 9.0342916e-01	 1.9449008e-01	[ 7.9984788e-01]	 1.8278575e-01


.. parsed-literal::

      33	 8.8576052e-01	 1.8111392e-01	 9.2802223e-01	 1.8929279e-01	[ 8.1987579e-01]	 2.0298910e-01


.. parsed-literal::

      34	 9.0166458e-01	 1.7885568e-01	 9.4461456e-01	 1.8688046e-01	[ 8.3580476e-01]	 2.0846057e-01
      35	 9.1394196e-01	 1.7800611e-01	 9.5715651e-01	 1.8644222e-01	[ 8.4824644e-01]	 1.9923067e-01


.. parsed-literal::

      36	 9.3154916e-01	 1.7559218e-01	 9.7496149e-01	 1.8470773e-01	[ 8.6750518e-01]	 1.7635179e-01


.. parsed-literal::

      37	 9.4838559e-01	 1.7165848e-01	 9.9225207e-01	 1.8049040e-01	[ 8.8717550e-01]	 2.0932364e-01
      38	 9.6355556e-01	 1.6998397e-01	 1.0077102e+00	 1.7910929e-01	[ 9.0381818e-01]	 1.8204451e-01


.. parsed-literal::

      39	 9.7163202e-01	 1.6943243e-01	 1.0161693e+00	 1.7843167e-01	[ 9.0853966e-01]	 1.9579005e-01
      40	 9.9491134e-01	 1.6817538e-01	 1.0412200e+00	 1.7647387e-01	[ 9.1937204e-01]	 1.7420721e-01


.. parsed-literal::

      41	 1.0029682e+00	 1.6939012e-01	 1.0509350e+00	 1.7645442e-01	[ 9.2917612e-01]	 1.9486094e-01
      42	 1.0261051e+00	 1.6636077e-01	 1.0735832e+00	 1.7428761e-01	[ 9.4988979e-01]	 1.6104388e-01


.. parsed-literal::

      43	 1.0356840e+00	 1.6528601e-01	 1.0830654e+00	 1.7378754e-01	[ 9.5927816e-01]	 2.0291376e-01
      44	 1.0485374e+00	 1.6355101e-01	 1.0964594e+00	 1.7250491e-01	[ 9.6381943e-01]	 1.7981339e-01


.. parsed-literal::

      45	 1.0625215e+00	 1.6127282e-01	 1.1111641e+00	 1.7088963e-01	[ 9.7265522e-01]	 1.6233993e-01
      46	 1.0745674e+00	 1.5867780e-01	 1.1234968e+00	 1.6842922e-01	[ 9.7857397e-01]	 1.8409061e-01


.. parsed-literal::

      47	 1.0851491e+00	 1.5556753e-01	 1.1343530e+00	 1.6576643e-01	[ 9.8208499e-01]	 1.9103789e-01
      48	 1.0943059e+00	 1.5308139e-01	 1.1438315e+00	 1.6373391e-01	[ 9.8963132e-01]	 1.6609764e-01


.. parsed-literal::

      49	 1.1005844e+00	 1.4951432e-01	 1.1507023e+00	 1.6104316e-01	  9.8377440e-01 	 2.0522904e-01
      50	 1.1127733e+00	 1.4886439e-01	 1.1623720e+00	 1.6034599e-01	[ 1.0064154e+00]	 1.7963672e-01


.. parsed-literal::

      51	 1.1183097e+00	 1.4846977e-01	 1.1678548e+00	 1.5995218e-01	[ 1.0149194e+00]	 1.6543555e-01
      52	 1.1274445e+00	 1.4770332e-01	 1.1769714e+00	 1.5933218e-01	[ 1.0262082e+00]	 1.6963124e-01


.. parsed-literal::

      53	 1.1399872e+00	 1.4617337e-01	 1.1897008e+00	 1.5813700e-01	[ 1.0453304e+00]	 1.9876719e-01
      54	 1.1491649e+00	 1.4496285e-01	 1.1988876e+00	 1.5767604e-01	[ 1.0575067e+00]	 1.9004273e-01


.. parsed-literal::

      55	 1.1572086e+00	 1.4426026e-01	 1.2068774e+00	 1.5700454e-01	[ 1.0662392e+00]	 2.0719790e-01
      56	 1.1664660e+00	 1.4291442e-01	 1.2166429e+00	 1.5643300e-01	[ 1.0734696e+00]	 1.9340491e-01


.. parsed-literal::

      57	 1.1743791e+00	 1.4194791e-01	 1.2247921e+00	 1.5619527e-01	[ 1.0796391e+00]	 1.9541955e-01
      58	 1.1851891e+00	 1.4041802e-01	 1.2365494e+00	 1.5680955e-01	[ 1.0827511e+00]	 1.9089556e-01


.. parsed-literal::

      59	 1.1948848e+00	 1.4064696e-01	 1.2464054e+00	 1.5801527e-01	[ 1.0894372e+00]	 1.8671823e-01
      60	 1.2019904e+00	 1.4023917e-01	 1.2533451e+00	 1.5794242e-01	[ 1.0915585e+00]	 1.8052411e-01


.. parsed-literal::

      61	 1.2131898e+00	 1.3987960e-01	 1.2649147e+00	 1.5917515e-01	[ 1.0921122e+00]	 1.5996027e-01


.. parsed-literal::

      62	 1.2211609e+00	 1.3905193e-01	 1.2733457e+00	 1.5897161e-01	[ 1.0934295e+00]	 2.0427132e-01


.. parsed-literal::

      63	 1.2295145e+00	 1.3818593e-01	 1.2817851e+00	 1.5870586e-01	[ 1.1046834e+00]	 2.0861101e-01
      64	 1.2375339e+00	 1.3694537e-01	 1.2897926e+00	 1.5762338e-01	[ 1.1173994e+00]	 1.8090606e-01


.. parsed-literal::

      65	 1.2435182e+00	 1.3621471e-01	 1.2962628e+00	 1.5683859e-01	[ 1.1181580e+00]	 1.6778159e-01


.. parsed-literal::

      66	 1.2506484e+00	 1.3547168e-01	 1.3032320e+00	 1.5562413e-01	[ 1.1232010e+00]	 2.0350766e-01
      67	 1.2554594e+00	 1.3518757e-01	 1.3082175e+00	 1.5577897e-01	  1.1206727e+00 	 1.7877603e-01


.. parsed-literal::

      68	 1.2631741e+00	 1.3462238e-01	 1.3160657e+00	 1.5529637e-01	  1.1168974e+00 	 1.9167399e-01
      69	 1.2688143e+00	 1.3410583e-01	 1.3221613e+00	 1.5610849e-01	  1.1052574e+00 	 1.7386389e-01


.. parsed-literal::

      70	 1.2750472e+00	 1.3374944e-01	 1.3282846e+00	 1.5531336e-01	  1.1086509e+00 	 1.7202139e-01


.. parsed-literal::

      71	 1.2784518e+00	 1.3351318e-01	 1.3315930e+00	 1.5519355e-01	  1.1142134e+00 	 2.0798707e-01
      72	 1.2850504e+00	 1.3308653e-01	 1.3385014e+00	 1.5527728e-01	  1.1101219e+00 	 1.6639876e-01


.. parsed-literal::

      73	 1.2903157e+00	 1.3269302e-01	 1.3443196e+00	 1.5511912e-01	  1.1011422e+00 	 1.9781041e-01
      74	 1.2957312e+00	 1.3241571e-01	 1.3496958e+00	 1.5488570e-01	  1.1013092e+00 	 1.9001770e-01


.. parsed-literal::

      75	 1.3014739e+00	 1.3215838e-01	 1.3555571e+00	 1.5428610e-01	  1.0956023e+00 	 1.9865322e-01


.. parsed-literal::

      76	 1.3050973e+00	 1.3181725e-01	 1.3592522e+00	 1.5380044e-01	  1.0974056e+00 	 2.0450234e-01
      77	 1.3107412e+00	 1.3161944e-01	 1.3650791e+00	 1.5289421e-01	  1.0967365e+00 	 1.7548633e-01


.. parsed-literal::

      78	 1.3159157e+00	 1.3139534e-01	 1.3702542e+00	 1.5253751e-01	  1.1054994e+00 	 1.9534492e-01


.. parsed-literal::

      79	 1.3204535e+00	 1.3114458e-01	 1.3747113e+00	 1.5222400e-01	  1.1137140e+00 	 2.0877647e-01
      80	 1.3273198e+00	 1.3082525e-01	 1.3815356e+00	 1.5168917e-01	[ 1.1279462e+00]	 1.9398117e-01


.. parsed-literal::

      81	 1.3324234e+00	 1.3048530e-01	 1.3865507e+00	 1.5162789e-01	[ 1.1411171e+00]	 1.9116640e-01


.. parsed-literal::

      82	 1.3366711e+00	 1.3005756e-01	 1.3907602e+00	 1.5120050e-01	[ 1.1468875e+00]	 2.1148086e-01


.. parsed-literal::

      83	 1.3414369e+00	 1.2966145e-01	 1.3955742e+00	 1.5085756e-01	[ 1.1515581e+00]	 2.0671844e-01
      84	 1.3459692e+00	 1.2920354e-01	 1.4002979e+00	 1.5053641e-01	  1.1485462e+00 	 1.8092084e-01


.. parsed-literal::

      85	 1.3503866e+00	 1.2837857e-01	 1.4051125e+00	 1.4951726e-01	  1.1322612e+00 	 1.9800091e-01


.. parsed-literal::

      86	 1.3563374e+00	 1.2793805e-01	 1.4110330e+00	 1.4927480e-01	  1.1392818e+00 	 2.1226239e-01


.. parsed-literal::

      87	 1.3593983e+00	 1.2768373e-01	 1.4140646e+00	 1.4898561e-01	  1.1393871e+00 	 2.0727134e-01


.. parsed-literal::

      88	 1.3646658e+00	 1.2726691e-01	 1.4196075e+00	 1.4836695e-01	  1.1396787e+00 	 2.1123028e-01
      89	 1.3676837e+00	 1.2665557e-01	 1.4232197e+00	 1.4736428e-01	  1.1326163e+00 	 1.9889569e-01


.. parsed-literal::

      90	 1.3728411e+00	 1.2659880e-01	 1.4282476e+00	 1.4739742e-01	  1.1350360e+00 	 2.0098972e-01
      91	 1.3756792e+00	 1.2648285e-01	 1.4311129e+00	 1.4731683e-01	  1.1348360e+00 	 1.7805505e-01


.. parsed-literal::

      92	 1.3789647e+00	 1.2630793e-01	 1.4345395e+00	 1.4718722e-01	  1.1290720e+00 	 2.1063304e-01


.. parsed-literal::

      93	 1.3805515e+00	 1.2602492e-01	 1.4365032e+00	 1.4675778e-01	  1.1133110e+00 	 2.0850182e-01
      94	 1.3857967e+00	 1.2579766e-01	 1.4416966e+00	 1.4681880e-01	  1.1115186e+00 	 1.9258642e-01


.. parsed-literal::

      95	 1.3878857e+00	 1.2570242e-01	 1.4437469e+00	 1.4680333e-01	  1.1106828e+00 	 2.0588636e-01
      96	 1.3907066e+00	 1.2556763e-01	 1.4466602e+00	 1.4680145e-01	  1.1078516e+00 	 1.9365907e-01


.. parsed-literal::

      97	 1.3945751e+00	 1.2538555e-01	 1.4505954e+00	 1.4656307e-01	  1.1013640e+00 	 1.7464876e-01
      98	 1.3983361e+00	 1.2532056e-01	 1.4545822e+00	 1.4682734e-01	  1.1042413e+00 	 1.6585135e-01


.. parsed-literal::

      99	 1.4014016e+00	 1.2509758e-01	 1.4575608e+00	 1.4638314e-01	  1.1073413e+00 	 2.0361805e-01


.. parsed-literal::

     100	 1.4050907e+00	 1.2480130e-01	 1.4613008e+00	 1.4586008e-01	  1.1078301e+00 	 2.0112014e-01
     101	 1.4087015e+00	 1.2451507e-01	 1.4649628e+00	 1.4542400e-01	  1.1102170e+00 	 1.9346571e-01


.. parsed-literal::

     102	 1.4146283e+00	 1.2415211e-01	 1.4711343e+00	 1.4502059e-01	  1.0989381e+00 	 1.9825196e-01


.. parsed-literal::

     103	 1.4192392e+00	 1.2419033e-01	 1.4757731e+00	 1.4533921e-01	  1.1062092e+00 	 2.0519114e-01
     104	 1.4225414e+00	 1.2412473e-01	 1.4790030e+00	 1.4548311e-01	  1.1055896e+00 	 1.7986870e-01


.. parsed-literal::

     105	 1.4260540e+00	 1.2408987e-01	 1.4825798e+00	 1.4569611e-01	  1.1052119e+00 	 1.6719723e-01


.. parsed-literal::

     106	 1.4284913e+00	 1.2409646e-01	 1.4851089e+00	 1.4583921e-01	  1.1028587e+00 	 2.0963192e-01
     107	 1.4312148e+00	 1.2406510e-01	 1.4878227e+00	 1.4573333e-01	  1.1055712e+00 	 1.9250154e-01


.. parsed-literal::

     108	 1.4340479e+00	 1.2409887e-01	 1.4907174e+00	 1.4567474e-01	  1.1078942e+00 	 2.0055914e-01
     109	 1.4364083e+00	 1.2412819e-01	 1.4930294e+00	 1.4573625e-01	  1.1107885e+00 	 1.9932914e-01


.. parsed-literal::

     110	 1.4408504e+00	 1.2407860e-01	 1.4974527e+00	 1.4585735e-01	  1.1064375e+00 	 2.0085835e-01
     111	 1.4436894e+00	 1.2400220e-01	 1.5002240e+00	 1.4590353e-01	  1.1163229e+00 	 1.8203568e-01


.. parsed-literal::

     112	 1.4464429e+00	 1.2382874e-01	 1.5028980e+00	 1.4569780e-01	  1.1148911e+00 	 2.0672894e-01
     113	 1.4487112e+00	 1.2362191e-01	 1.5051811e+00	 1.4544459e-01	  1.1121551e+00 	 1.7359948e-01


.. parsed-literal::

     114	 1.4519552e+00	 1.2333580e-01	 1.5085452e+00	 1.4501872e-01	  1.1076769e+00 	 1.9807172e-01


.. parsed-literal::

     115	 1.4541843e+00	 1.2307425e-01	 1.5110688e+00	 1.4457057e-01	  1.1094363e+00 	 2.0313597e-01


.. parsed-literal::

     116	 1.4575567e+00	 1.2298041e-01	 1.5143887e+00	 1.4445790e-01	  1.1080960e+00 	 2.0796704e-01
     117	 1.4594061e+00	 1.2293365e-01	 1.5162341e+00	 1.4443839e-01	  1.1081282e+00 	 1.9098258e-01


.. parsed-literal::

     118	 1.4620823e+00	 1.2278985e-01	 1.5189609e+00	 1.4439718e-01	  1.1022919e+00 	 1.7022204e-01
     119	 1.4655418e+00	 1.2269171e-01	 1.5225887e+00	 1.4434321e-01	  1.0825650e+00 	 1.9465399e-01


.. parsed-literal::

     120	 1.4680971e+00	 1.2243363e-01	 1.5253695e+00	 1.4419506e-01	  1.0582706e+00 	 1.9810009e-01
     121	 1.4703179e+00	 1.2238915e-01	 1.5274988e+00	 1.4404831e-01	  1.0599022e+00 	 1.7683673e-01


.. parsed-literal::

     122	 1.4723134e+00	 1.2229716e-01	 1.5295752e+00	 1.4382491e-01	  1.0548443e+00 	 1.9097590e-01
     123	 1.4744474e+00	 1.2216731e-01	 1.5318021e+00	 1.4357701e-01	  1.0484765e+00 	 1.6273260e-01


.. parsed-literal::

     124	 1.4761836e+00	 1.2212167e-01	 1.5336260e+00	 1.4347861e-01	  1.0430466e+00 	 3.0120111e-01
     125	 1.4785741e+00	 1.2205293e-01	 1.5360672e+00	 1.4338939e-01	  1.0372730e+00 	 1.9330907e-01


.. parsed-literal::

     126	 1.4804308e+00	 1.2203686e-01	 1.5378759e+00	 1.4347014e-01	  1.0345888e+00 	 1.9654417e-01
     127	 1.4826938e+00	 1.2201859e-01	 1.5400599e+00	 1.4360656e-01	  1.0329490e+00 	 1.9448996e-01


.. parsed-literal::

     128	 1.4850174e+00	 1.2202577e-01	 1.5423595e+00	 1.4383386e-01	  1.0218273e+00 	 1.9362450e-01


.. parsed-literal::

     129	 1.4871166e+00	 1.2192711e-01	 1.5444648e+00	 1.4373611e-01	  1.0205190e+00 	 2.0689106e-01
     130	 1.4889445e+00	 1.2178386e-01	 1.5463667e+00	 1.4345823e-01	  1.0169019e+00 	 1.8306160e-01


.. parsed-literal::

     131	 1.4912951e+00	 1.2157537e-01	 1.5488839e+00	 1.4303171e-01	  1.0125148e+00 	 1.9108081e-01
     132	 1.4930763e+00	 1.2123355e-01	 1.5508216e+00	 1.4228782e-01	  1.0112013e+00 	 1.9637322e-01


.. parsed-literal::

     133	 1.4946345e+00	 1.2117674e-01	 1.5523587e+00	 1.4224974e-01	  1.0108705e+00 	 1.7863989e-01
     134	 1.4960106e+00	 1.2112593e-01	 1.5536785e+00	 1.4222493e-01	  1.0117992e+00 	 1.8152547e-01


.. parsed-literal::

     135	 1.4977793e+00	 1.2096613e-01	 1.5554164e+00	 1.4201496e-01	  1.0091040e+00 	 1.9686460e-01
     136	 1.4988725e+00	 1.2100608e-01	 1.5566133e+00	 1.4213693e-01	  1.0027927e+00 	 1.7593193e-01


.. parsed-literal::

     137	 1.5012821e+00	 1.2074748e-01	 1.5589694e+00	 1.4174597e-01	  1.0019613e+00 	 1.8103027e-01


.. parsed-literal::

     138	 1.5022779e+00	 1.2064881e-01	 1.5600230e+00	 1.4160811e-01	  9.9947538e-01 	 2.0238805e-01
     139	 1.5040260e+00	 1.2045226e-01	 1.5618992e+00	 1.4135655e-01	  9.9695888e-01 	 1.9907737e-01


.. parsed-literal::

     140	 1.5055095e+00	 1.2020028e-01	 1.5635646e+00	 1.4119814e-01	  9.8630673e-01 	 1.8508863e-01


.. parsed-literal::

     141	 1.5074918e+00	 1.2004433e-01	 1.5655732e+00	 1.4097030e-01	  9.9195938e-01 	 2.0989561e-01
     142	 1.5088217e+00	 1.1992750e-01	 1.5668567e+00	 1.4082808e-01	  9.9740167e-01 	 1.9780636e-01


.. parsed-literal::

     143	 1.5104675e+00	 1.1972985e-01	 1.5684455e+00	 1.4062157e-01	  9.9879691e-01 	 2.0502234e-01


.. parsed-literal::

     144	 1.5114911e+00	 1.1960464e-01	 1.5694556e+00	 1.4044026e-01	  9.9749354e-01 	 3.0918670e-01
     145	 1.5129434e+00	 1.1943674e-01	 1.5708812e+00	 1.4025424e-01	  9.9020326e-01 	 1.8198848e-01


.. parsed-literal::

     146	 1.5143937e+00	 1.1925488e-01	 1.5723362e+00	 1.3995655e-01	  9.7908692e-01 	 1.8012619e-01
     147	 1.5161790e+00	 1.1900062e-01	 1.5742625e+00	 1.3946241e-01	  9.5370353e-01 	 1.9784236e-01


.. parsed-literal::

     148	 1.5180630e+00	 1.1873291e-01	 1.5762586e+00	 1.3889458e-01	  9.3494534e-01 	 1.9152212e-01
     149	 1.5196223e+00	 1.1852285e-01	 1.5779075e+00	 1.3853021e-01	  9.2422818e-01 	 1.9870925e-01


.. parsed-literal::

     150	 1.5214655e+00	 1.1822681e-01	 1.5799099e+00	 1.3815068e-01	  9.0708186e-01 	 1.7032552e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 1min 58s, sys: 988 ms, total: 1min 59s
    Wall time: 29.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f0c0f039490>



This should have taken about 30 seconds on a typical desktop computer,
and you should now see a file called ``GPz_model.pkl`` in the directory.
This model file is used by the ``GPzEstimator`` stage to determine our
redshift PDFs for the test set of galaxies. Let’s set up that stage,
again defining a dictionary of variables for the config params:

.. code:: ipython3

    gpz_test_dict = dict(hdf5_groupname="photometry", model="GPz_model.pkl")
    
    gpz_run = GPzEstimator.make_stage(name="gpz_run", **gpz_test_dict)

Let’s run the stage and compute photo-z’s for our test set:

.. code:: ipython3

    %%time
    results = gpz_run.estimate(test_data)


.. parsed-literal::

    Inserting handle into data store.  input: None, gpz_run
    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 20,449
    Process 0 estimating GPz PZ PDF for rows 0 - 20,449


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run
    CPU times: user 934 ms, sys: 47 ms, total: 981 ms
    Wall time: 352 ms


This should be very fast, under a second for our 20,449 galaxies in the
test set. Now, let’s plot a scatter plot of the point estimates, as well
as a few example PDFs. We can get access to the ``qp`` ensemble that was
written via the DataStore via ``results()``

.. code:: ipython3

    ens = results()

.. code:: ipython3

    expdfids = [2, 180, 13517, 18032]
    fig, axs = plt.subplots(4, 1, figsize=(12,10))
    for i, xx in enumerate(expdfids):
        axs[i].set_xlim(0,3)
        ens[xx].plot_native(axes=axs[i])
    axs[3].set_xlabel("redshift", fontsize=15)




.. parsed-literal::

    Text(0.5, 0, 'redshift')




.. image:: 06_GPz_files/06_GPz_15_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: 06_GPz_files/06_GPz_18_1.png

