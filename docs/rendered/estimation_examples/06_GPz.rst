GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

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
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # set up the DataStore to keep track of data
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = DS.read_file("training_data", TableHandle, trainFile)
    test_data = DS.read_file("test_data", TableHandle, testFile)

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

    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4072003e-01	 3.1994972e-01	-3.3101752e-01	 3.2348521e-01	[-3.3677615e-01]	 4.6394563e-01


.. parsed-literal::

       2	-2.7015043e-01	 3.0896675e-01	-2.4579415e-01	 3.1376805e-01	[-2.5778834e-01]	 2.3273516e-01


.. parsed-literal::

       3	-2.2628578e-01	 2.8848536e-01	-1.8379970e-01	 2.9548348e-01	[-2.0725701e-01]	 2.8496504e-01
       4	-1.9132077e-01	 2.6429670e-01	-1.4911151e-01	 2.7038871e-01	[-1.8545705e-01]	 1.7342973e-01


.. parsed-literal::

       5	-1.0211455e-01	 2.5594888e-01	-6.7552744e-02	 2.6807701e-01	[-1.1547901e-01]	 2.1017265e-01


.. parsed-literal::

       6	-6.5161473e-02	 2.5000725e-01	-3.4016418e-02	 2.5838872e-01	[-6.5117852e-02]	 2.0371795e-01


.. parsed-literal::

       7	-4.8727586e-02	 2.4768761e-01	-2.3930473e-02	 2.5691743e-01	[-5.8852620e-02]	 2.0821476e-01


.. parsed-literal::

       8	-3.4944865e-02	 2.4542710e-01	-1.4460215e-02	 2.5464108e-01	[-4.9158226e-02]	 2.0723534e-01
       9	-2.1827454e-02	 2.4308357e-01	-4.3739163e-03	 2.5167987e-01	[-3.7652558e-02]	 1.8160701e-01


.. parsed-literal::

      10	-1.1619711e-02	 2.4140267e-01	 3.5108009e-03	 2.4855811e-01	[-2.4362113e-02]	 2.0619130e-01


.. parsed-literal::

      11	-5.6268504e-03	 2.4028296e-01	 8.5319941e-03	 2.4688478e-01	[-1.8294635e-02]	 2.1445990e-01
      12	-3.3850531e-03	 2.3980274e-01	 1.0638785e-02	 2.4654982e-01	[-1.6187023e-02]	 1.9625878e-01


.. parsed-literal::

      13	 2.0764237e-03	 2.3871010e-01	 1.6195895e-02	 2.4591241e-01	[-1.0670003e-02]	 2.0295334e-01


.. parsed-literal::

      14	 1.1268366e-01	 2.2430311e-01	 1.3330896e-01	 2.3200227e-01	[ 1.2990257e-01]	 4.3635392e-01


.. parsed-literal::

      15	 2.2262270e-01	 2.2216551e-01	 2.5054295e-01	 2.2343569e-01	[ 2.5420259e-01]	 2.1531343e-01
      16	 2.6236772e-01	 2.1880995e-01	 2.9269944e-01	 2.2096603e-01	[ 2.8370179e-01]	 1.9835258e-01


.. parsed-literal::

      17	 3.1162268e-01	 2.1525490e-01	 3.4315514e-01	 2.1706705e-01	[ 3.3064234e-01]	 1.8635607e-01


.. parsed-literal::

      18	 3.7418372e-01	 2.1506213e-01	 4.0576503e-01	 2.1485768e-01	[ 4.0149576e-01]	 2.1255565e-01


.. parsed-literal::

      19	 4.3344550e-01	 2.1104517e-01	 4.6616314e-01	 2.1067619e-01	[ 4.5995760e-01]	 2.0669436e-01


.. parsed-literal::

      20	 5.2479667e-01	 2.1048184e-01	 5.6171541e-01	 2.0945274e-01	[ 5.4910028e-01]	 2.0977283e-01


.. parsed-literal::

      21	 5.7108722e-01	 2.0948038e-01	 6.1031319e-01	 2.0785929e-01	[ 5.7874575e-01]	 2.0824981e-01


.. parsed-literal::

      22	 5.9255773e-01	 2.0668412e-01	 6.2964370e-01	 2.0465663e-01	[ 6.0792788e-01]	 2.2077847e-01
      23	 6.2560626e-01	 2.0496597e-01	 6.6340288e-01	 2.0345518e-01	[ 6.3605396e-01]	 1.9966125e-01


.. parsed-literal::

      24	 6.5585763e-01	 2.0668838e-01	 6.9356327e-01	 2.0688913e-01	[ 6.6608577e-01]	 2.0403814e-01
      25	 6.8843189e-01	 2.0378855e-01	 7.2570514e-01	 2.0514974e-01	[ 6.9338313e-01]	 1.9188166e-01


.. parsed-literal::

      26	 7.2272450e-01	 2.0203134e-01	 7.5993539e-01	 2.0344999e-01	[ 7.2922247e-01]	 2.1357608e-01


.. parsed-literal::

      27	 7.5109337e-01	 2.0206413e-01	 7.8928514e-01	 2.0629297e-01	[ 7.5756826e-01]	 2.1475053e-01


.. parsed-literal::

      28	 7.7473320e-01	 2.0366049e-01	 8.1448768e-01	 2.0599956e-01	[ 7.9056031e-01]	 2.1582913e-01


.. parsed-literal::

      29	 7.7850129e-01	 2.0680454e-01	 8.1863074e-01	 2.0751523e-01	[ 8.1418384e-01]	 2.1250200e-01


.. parsed-literal::

      30	 8.1379772e-01	 2.0872055e-01	 8.5592855e-01	 2.1130379e-01	[ 8.4107120e-01]	 2.1808028e-01


.. parsed-literal::

      31	 8.3578616e-01	 2.0725744e-01	 8.7823827e-01	 2.0963878e-01	[ 8.6452596e-01]	 2.1420288e-01


.. parsed-literal::

      32	 8.6264162e-01	 2.1052221e-01	 9.0574768e-01	 2.1407059e-01	[ 8.9484629e-01]	 2.0778441e-01
      33	 8.8821314e-01	 2.1166981e-01	 9.3105401e-01	 2.1524143e-01	[ 9.1971647e-01]	 1.7140102e-01


.. parsed-literal::

      34	 9.1319887e-01	 2.1202306e-01	 9.5663061e-01	 2.1586174e-01	[ 9.4042464e-01]	 1.8307829e-01
      35	 9.3960533e-01	 2.0757201e-01	 9.8403567e-01	 2.1133367e-01	[ 9.7125714e-01]	 1.9775128e-01


.. parsed-literal::

      36	 9.5417166e-01	 2.0060036e-01	 1.0000921e+00	 2.0393287e-01	[ 9.8576557e-01]	 1.7932367e-01
      37	 9.6730541e-01	 1.9867982e-01	 1.0127847e+00	 2.0106336e-01	[ 9.9723938e-01]	 1.9585323e-01


.. parsed-literal::

      38	 9.7952387e-01	 1.9544608e-01	 1.0252105e+00	 1.9654259e-01	[ 1.0040769e+00]	 1.6940665e-01


.. parsed-literal::

      39	 9.9655204e-01	 1.9169220e-01	 1.0425142e+00	 1.9122144e-01	[ 1.0160555e+00]	 2.0890617e-01
      40	 1.0155649e+00	 1.8673068e-01	 1.0627317e+00	 1.8616207e-01	[ 1.0186547e+00]	 1.9034028e-01


.. parsed-literal::

      41	 1.0342573e+00	 1.8485072e-01	 1.0814574e+00	 1.8466756e-01	[ 1.0318667e+00]	 2.0838213e-01
      42	 1.0471521e+00	 1.8419697e-01	 1.0941343e+00	 1.8406650e-01	[ 1.0443877e+00]	 1.9559908e-01


.. parsed-literal::

      43	 1.0627970e+00	 1.8120023e-01	 1.1101502e+00	 1.8181970e-01	[ 1.0578213e+00]	 2.0934129e-01
      44	 1.0735137e+00	 1.7859894e-01	 1.1212402e+00	 1.7802112e-01	[ 1.0722676e+00]	 1.9079757e-01


.. parsed-literal::

      45	 1.0834432e+00	 1.7576000e-01	 1.1310550e+00	 1.7516373e-01	[ 1.0825893e+00]	 2.1105409e-01
      46	 1.0962106e+00	 1.7201415e-01	 1.1439309e+00	 1.7069429e-01	[ 1.0919056e+00]	 1.9916844e-01


.. parsed-literal::

      47	 1.1067900e+00	 1.6916356e-01	 1.1546957e+00	 1.6722831e-01	[ 1.1001845e+00]	 2.0127678e-01
      48	 1.1202497e+00	 1.6585817e-01	 1.1688174e+00	 1.6386806e-01	[ 1.1135244e+00]	 1.8742776e-01


.. parsed-literal::

      49	 1.1334786e+00	 1.6347477e-01	 1.1820831e+00	 1.6214683e-01	[ 1.1258516e+00]	 2.0299077e-01


.. parsed-literal::

      50	 1.1461576e+00	 1.6150764e-01	 1.1951064e+00	 1.6107558e-01	[ 1.1314725e+00]	 2.1943164e-01


.. parsed-literal::

      51	 1.1616052e+00	 1.5571022e-01	 1.2117815e+00	 1.5971072e-01	  1.1251924e+00 	 2.0692563e-01


.. parsed-literal::

      52	 1.1714392e+00	 1.5607063e-01	 1.2220935e+00	 1.6103176e-01	  1.0974784e+00 	 2.1429396e-01


.. parsed-literal::

      53	 1.1809259e+00	 1.5422741e-01	 1.2312491e+00	 1.5974667e-01	  1.1128534e+00 	 2.1316123e-01


.. parsed-literal::

      54	 1.1955742e+00	 1.5137967e-01	 1.2463833e+00	 1.5850433e-01	  1.1135869e+00 	 2.0707560e-01


.. parsed-literal::

      55	 1.2092098e+00	 1.4959665e-01	 1.2598343e+00	 1.5721528e-01	  1.1164010e+00 	 2.1517181e-01
      56	 1.2226665e+00	 1.4742171e-01	 1.2746991e+00	 1.5587561e-01	  1.0874462e+00 	 1.7280221e-01


.. parsed-literal::

      57	 1.2340075e+00	 1.4703849e-01	 1.2853078e+00	 1.5451350e-01	  1.0996440e+00 	 2.1409798e-01
      58	 1.2428748e+00	 1.4647585e-01	 1.2938889e+00	 1.5403183e-01	  1.1108291e+00 	 1.8161917e-01


.. parsed-literal::

      59	 1.2557337e+00	 1.4521388e-01	 1.3068585e+00	 1.5383806e-01	  1.1180829e+00 	 2.0852757e-01
      60	 1.2631258e+00	 1.4553460e-01	 1.3143377e+00	 1.5316349e-01	  1.1190690e+00 	 1.8385553e-01


.. parsed-literal::

      61	 1.2721365e+00	 1.4432332e-01	 1.3232944e+00	 1.5365232e-01	  1.1234276e+00 	 1.8711472e-01


.. parsed-literal::

      62	 1.2795923e+00	 1.4357744e-01	 1.3311239e+00	 1.5470441e-01	  1.1188042e+00 	 2.0788217e-01
      63	 1.2868074e+00	 1.4307955e-01	 1.3384561e+00	 1.5505358e-01	  1.1169090e+00 	 1.8332386e-01


.. parsed-literal::

      64	 1.3001144e+00	 1.4148036e-01	 1.3525093e+00	 1.5486309e-01	  1.0992175e+00 	 2.0268822e-01


.. parsed-literal::

      65	 1.3077976e+00	 1.4021559e-01	 1.3598748e+00	 1.5257087e-01	  1.1053385e+00 	 2.0421767e-01


.. parsed-literal::

      66	 1.3151161e+00	 1.3953929e-01	 1.3669523e+00	 1.5126701e-01	  1.1033865e+00 	 2.1118927e-01


.. parsed-literal::

      67	 1.3204320e+00	 1.3875428e-01	 1.3724576e+00	 1.4977486e-01	  1.0916128e+00 	 2.1416354e-01


.. parsed-literal::

      68	 1.3264536e+00	 1.3781888e-01	 1.3786020e+00	 1.4778488e-01	  1.0745764e+00 	 2.1559811e-01


.. parsed-literal::

      69	 1.3318850e+00	 1.3628759e-01	 1.3839852e+00	 1.4609334e-01	  1.0548652e+00 	 2.1989202e-01


.. parsed-literal::

      70	 1.3407418e+00	 1.3589506e-01	 1.3928330e+00	 1.4506078e-01	  1.0369621e+00 	 2.0192122e-01


.. parsed-literal::

      71	 1.3446796e+00	 1.3563047e-01	 1.3967969e+00	 1.4524155e-01	  1.0385732e+00 	 2.0895290e-01


.. parsed-literal::

      72	 1.3516246e+00	 1.3489304e-01	 1.4039672e+00	 1.4502047e-01	  1.0276309e+00 	 2.1860862e-01


.. parsed-literal::

      73	 1.3573382e+00	 1.3424444e-01	 1.4098132e+00	 1.4507215e-01	  1.0165583e+00 	 2.1130371e-01
      74	 1.3646207e+00	 1.3347339e-01	 1.4171622e+00	 1.4422248e-01	  1.0073828e+00 	 1.9887877e-01


.. parsed-literal::

      75	 1.3692906e+00	 1.3308293e-01	 1.4218734e+00	 1.4358870e-01	  1.0056321e+00 	 1.9917297e-01


.. parsed-literal::

      76	 1.3744112e+00	 1.3251040e-01	 1.4271161e+00	 1.4330322e-01	  1.0167936e+00 	 2.1012592e-01


.. parsed-literal::

      77	 1.3809227e+00	 1.3175072e-01	 1.4336303e+00	 1.4394993e-01	  1.0587008e+00 	 2.0371032e-01


.. parsed-literal::

      78	 1.3862505e+00	 1.3103520e-01	 1.4389041e+00	 1.4480192e-01	[ 1.1347121e+00]	 2.0512462e-01


.. parsed-literal::

      79	 1.3906557e+00	 1.3094486e-01	 1.4432582e+00	 1.4509996e-01	  1.1346592e+00 	 2.1146798e-01


.. parsed-literal::

      80	 1.3949962e+00	 1.3069166e-01	 1.4477834e+00	 1.4550382e-01	[ 1.1475516e+00]	 2.1304226e-01
      81	 1.3993541e+00	 1.3044747e-01	 1.4521564e+00	 1.4581917e-01	[ 1.1712564e+00]	 2.0329618e-01


.. parsed-literal::

      82	 1.4040688e+00	 1.2988829e-01	 1.4570626e+00	 1.4558786e-01	[ 1.1943308e+00]	 2.0581508e-01


.. parsed-literal::

      83	 1.4087335e+00	 1.2935209e-01	 1.4617507e+00	 1.4601693e-01	[ 1.2112056e+00]	 2.0793533e-01


.. parsed-literal::

      84	 1.4123659e+00	 1.2907596e-01	 1.4654127e+00	 1.4581037e-01	  1.2101931e+00 	 2.0942354e-01


.. parsed-literal::

      85	 1.4170946e+00	 1.2863285e-01	 1.4702387e+00	 1.4592660e-01	  1.2097127e+00 	 2.0198417e-01
      86	 1.4198220e+00	 1.2804481e-01	 1.4735357e+00	 1.4623950e-01	[ 1.2172096e+00]	 2.0425296e-01


.. parsed-literal::

      87	 1.4257482e+00	 1.2794204e-01	 1.4791765e+00	 1.4587098e-01	  1.2085506e+00 	 2.0385361e-01
      88	 1.4280924e+00	 1.2787903e-01	 1.4814467e+00	 1.4602492e-01	  1.2168893e+00 	 1.9893575e-01


.. parsed-literal::

      89	 1.4314010e+00	 1.2778562e-01	 1.4847282e+00	 1.4602787e-01	[ 1.2218291e+00]	 2.0088291e-01
      90	 1.4365606e+00	 1.2727963e-01	 1.4899792e+00	 1.4548675e-01	  1.2175181e+00 	 1.9438577e-01


.. parsed-literal::

      91	 1.4400282e+00	 1.2714765e-01	 1.4936191e+00	 1.4498992e-01	  1.2137536e+00 	 2.0681667e-01


.. parsed-literal::

      92	 1.4432499e+00	 1.2663552e-01	 1.4968416e+00	 1.4475449e-01	  1.2094683e+00 	 2.0669794e-01


.. parsed-literal::

      93	 1.4451919e+00	 1.2636560e-01	 1.4988596e+00	 1.4482391e-01	  1.2110557e+00 	 2.1109152e-01


.. parsed-literal::

      94	 1.4483977e+00	 1.2606444e-01	 1.5022075e+00	 1.4502165e-01	  1.2094881e+00 	 2.1833920e-01


.. parsed-literal::

      95	 1.4509516e+00	 1.2601640e-01	 1.5049903e+00	 1.4578085e-01	  1.2175104e+00 	 2.1204853e-01


.. parsed-literal::

      96	 1.4541366e+00	 1.2596512e-01	 1.5081434e+00	 1.4592280e-01	  1.2201749e+00 	 2.0543337e-01


.. parsed-literal::

      97	 1.4556824e+00	 1.2595104e-01	 1.5096456e+00	 1.4592371e-01	  1.2196707e+00 	 2.1211314e-01


.. parsed-literal::

      98	 1.4588653e+00	 1.2589074e-01	 1.5129506e+00	 1.4624195e-01	  1.2156274e+00 	 2.1423793e-01
      99	 1.4624038e+00	 1.2588936e-01	 1.5164989e+00	 1.4648816e-01	  1.2116619e+00 	 1.8241191e-01


.. parsed-literal::

     100	 1.4649664e+00	 1.2592762e-01	 1.5191242e+00	 1.4685001e-01	  1.2137823e+00 	 1.9623446e-01
     101	 1.4673504e+00	 1.2586352e-01	 1.5214790e+00	 1.4688528e-01	  1.2202866e+00 	 1.7746639e-01


.. parsed-literal::

     102	 1.4703442e+00	 1.2577445e-01	 1.5244619e+00	 1.4685521e-01	[ 1.2273233e+00]	 2.0773792e-01
     103	 1.4726633e+00	 1.2581798e-01	 1.5268236e+00	 1.4708045e-01	[ 1.2324035e+00]	 1.7884827e-01


.. parsed-literal::

     104	 1.4758346e+00	 1.2578563e-01	 1.5300158e+00	 1.4705045e-01	[ 1.2368000e+00]	 1.9838166e-01


.. parsed-literal::

     105	 1.4780999e+00	 1.2563224e-01	 1.5322024e+00	 1.4659834e-01	[ 1.2384571e+00]	 2.0273232e-01


.. parsed-literal::

     106	 1.4805167e+00	 1.2545377e-01	 1.5346031e+00	 1.4691538e-01	[ 1.2390574e+00]	 2.1315837e-01
     107	 1.4829874e+00	 1.2528324e-01	 1.5370915e+00	 1.4698425e-01	  1.2372216e+00 	 1.7730570e-01


.. parsed-literal::

     108	 1.4857115e+00	 1.2480353e-01	 1.5399647e+00	 1.4746956e-01	  1.2334737e+00 	 1.9020200e-01
     109	 1.4877050e+00	 1.2501680e-01	 1.5420057e+00	 1.4855982e-01	[ 1.2403401e+00]	 1.8932080e-01


.. parsed-literal::

     110	 1.4891987e+00	 1.2495421e-01	 1.5434242e+00	 1.4811510e-01	[ 1.2407005e+00]	 2.0422101e-01
     111	 1.4908148e+00	 1.2491906e-01	 1.5450379e+00	 1.4828094e-01	[ 1.2435503e+00]	 1.8722343e-01


.. parsed-literal::

     112	 1.4933055e+00	 1.2487542e-01	 1.5475747e+00	 1.4879174e-01	[ 1.2482616e+00]	 2.1562719e-01
     113	 1.4951335e+00	 1.2495482e-01	 1.5494516e+00	 1.4965613e-01	[ 1.2542253e+00]	 1.6848707e-01


.. parsed-literal::

     114	 1.4981996e+00	 1.2471312e-01	 1.5524715e+00	 1.4942926e-01	[ 1.2562422e+00]	 2.1349049e-01


.. parsed-literal::

     115	 1.5001230e+00	 1.2454874e-01	 1.5543935e+00	 1.4928494e-01	  1.2559927e+00 	 2.1004224e-01


.. parsed-literal::

     116	 1.5018271e+00	 1.2436623e-01	 1.5561790e+00	 1.4910951e-01	  1.2530330e+00 	 2.1781230e-01


.. parsed-literal::

     117	 1.5035196e+00	 1.2417584e-01	 1.5581733e+00	 1.4864592e-01	  1.2449945e+00 	 2.0905924e-01


.. parsed-literal::

     118	 1.5058471e+00	 1.2398546e-01	 1.5605435e+00	 1.4872795e-01	  1.2445167e+00 	 2.0902514e-01


.. parsed-literal::

     119	 1.5072055e+00	 1.2393827e-01	 1.5618938e+00	 1.4887383e-01	  1.2462283e+00 	 2.1163201e-01


.. parsed-literal::

     120	 1.5091015e+00	 1.2392668e-01	 1.5638363e+00	 1.4911148e-01	  1.2459864e+00 	 2.1859336e-01
     121	 1.5099178e+00	 1.2363271e-01	 1.5647958e+00	 1.4916106e-01	  1.2398338e+00 	 1.9842529e-01


.. parsed-literal::

     122	 1.5119386e+00	 1.2374672e-01	 1.5666850e+00	 1.4912693e-01	  1.2428458e+00 	 2.1218801e-01
     123	 1.5127665e+00	 1.2373669e-01	 1.5675151e+00	 1.4905913e-01	  1.2402248e+00 	 1.9315004e-01


.. parsed-literal::

     124	 1.5137732e+00	 1.2370467e-01	 1.5685314e+00	 1.4897159e-01	  1.2370727e+00 	 2.1685362e-01


.. parsed-literal::

     125	 1.5157167e+00	 1.2360200e-01	 1.5705623e+00	 1.4884488e-01	  1.2304470e+00 	 2.2345495e-01


.. parsed-literal::

     126	 1.5166075e+00	 1.2344389e-01	 1.5717090e+00	 1.4897293e-01	  1.2263741e+00 	 2.1843386e-01


.. parsed-literal::

     127	 1.5189081e+00	 1.2339014e-01	 1.5739105e+00	 1.4874503e-01	  1.2281818e+00 	 2.0847559e-01


.. parsed-literal::

     128	 1.5197490e+00	 1.2335053e-01	 1.5747384e+00	 1.4878141e-01	  1.2313265e+00 	 2.1324611e-01


.. parsed-literal::

     129	 1.5210051e+00	 1.2331311e-01	 1.5759940e+00	 1.4854637e-01	  1.2349792e+00 	 2.1331191e-01


.. parsed-literal::

     130	 1.5222985e+00	 1.2322747e-01	 1.5772900e+00	 1.4877702e-01	  1.2400866e+00 	 2.1549582e-01


.. parsed-literal::

     131	 1.5234871e+00	 1.2322516e-01	 1.5784122e+00	 1.4863607e-01	  1.2425282e+00 	 2.1677542e-01
     132	 1.5257143e+00	 1.2316158e-01	 1.5805337e+00	 1.4838315e-01	  1.2462840e+00 	 1.9018841e-01


.. parsed-literal::

     133	 1.5269556e+00	 1.2310350e-01	 1.5817538e+00	 1.4824281e-01	  1.2486581e+00 	 2.0439386e-01
     134	 1.5285276e+00	 1.2291726e-01	 1.5833369e+00	 1.4801580e-01	  1.2494554e+00 	 1.7564654e-01


.. parsed-literal::

     135	 1.5299471e+00	 1.2273676e-01	 1.5848267e+00	 1.4799796e-01	  1.2475444e+00 	 2.0671320e-01
     136	 1.5313600e+00	 1.2255407e-01	 1.5863080e+00	 1.4766364e-01	  1.2459665e+00 	 1.9462848e-01


.. parsed-literal::

     137	 1.5326975e+00	 1.2234059e-01	 1.5876729e+00	 1.4714456e-01	  1.2468333e+00 	 1.9637704e-01
     138	 1.5338339e+00	 1.2221959e-01	 1.5888443e+00	 1.4669363e-01	  1.2474094e+00 	 1.7501497e-01


.. parsed-literal::

     139	 1.5347096e+00	 1.2216894e-01	 1.5896682e+00	 1.4664248e-01	  1.2500038e+00 	 1.9385600e-01


.. parsed-literal::

     140	 1.5359114e+00	 1.2203561e-01	 1.5908305e+00	 1.4656051e-01	  1.2524718e+00 	 2.2151995e-01


.. parsed-literal::

     141	 1.5374504e+00	 1.2180751e-01	 1.5924195e+00	 1.4652343e-01	  1.2484903e+00 	 2.0897388e-01


.. parsed-literal::

     142	 1.5384183e+00	 1.2150141e-01	 1.5935363e+00	 1.4646587e-01	  1.2410453e+00 	 2.0313072e-01


.. parsed-literal::

     143	 1.5396756e+00	 1.2152592e-01	 1.5947867e+00	 1.4630818e-01	  1.2353596e+00 	 2.0817518e-01


.. parsed-literal::

     144	 1.5402719e+00	 1.2151745e-01	 1.5954211e+00	 1.4625963e-01	  1.2318724e+00 	 2.1146083e-01


.. parsed-literal::

     145	 1.5415586e+00	 1.2146566e-01	 1.5968173e+00	 1.4601796e-01	  1.2227867e+00 	 2.0889258e-01


.. parsed-literal::

     146	 1.5428703e+00	 1.2137610e-01	 1.5983102e+00	 1.4605442e-01	  1.2132475e+00 	 2.0309830e-01


.. parsed-literal::

     147	 1.5442357e+00	 1.2131772e-01	 1.5996899e+00	 1.4570552e-01	  1.2094840e+00 	 2.0116568e-01


.. parsed-literal::

     148	 1.5450914e+00	 1.2127431e-01	 1.6005324e+00	 1.4553557e-01	  1.2091107e+00 	 2.1537232e-01


.. parsed-literal::

     149	 1.5460178e+00	 1.2122427e-01	 1.6014629e+00	 1.4552258e-01	  1.2103093e+00 	 2.1115136e-01


.. parsed-literal::

     150	 1.5472363e+00	 1.2112975e-01	 1.6027777e+00	 1.4530234e-01	  1.2033485e+00 	 2.1760416e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.17 s, total: 2min 5s
    Wall time: 31.4 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fb738561510>



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

    Inserting handle into data store.  model: GPz_model.pkl, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.17 s, sys: 49.9 ms, total: 2.22 s
    Wall time: 683 ms


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




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_16_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data.data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_19_1.png

