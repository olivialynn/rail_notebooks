GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`GPz.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/estimation_examples/GPz.ipynb>`__
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
       1	-3.4483853e-01	 3.2079913e-01	-3.3455107e-01	 3.1961727e-01	[-3.3273215e-01]	 4.4949722e-01


.. parsed-literal::

       2	-2.7450275e-01	 3.1043339e-01	-2.5033658e-01	 3.1108637e-01	[-2.5269813e-01]	 2.2905469e-01


.. parsed-literal::

       3	-2.3235007e-01	 2.9080312e-01	-1.9126997e-01	 2.9158483e-01	[-1.9711882e-01]	 2.7344370e-01
       4	-1.8409170e-01	 2.6570235e-01	-1.3928586e-01	 2.6692524e-01	[-1.5001623e-01]	 1.6996145e-01


.. parsed-literal::

       5	-1.0605689e-01	 2.5619846e-01	-7.4668709e-02	 2.6110108e-01	[-1.0156900e-01]	 1.8985868e-01
       6	-6.7817451e-02	 2.5080332e-01	-3.8347139e-02	 2.5526604e-01	[-5.4970011e-02]	 1.7497921e-01


.. parsed-literal::

       7	-5.1745095e-02	 2.4849117e-01	-2.6924700e-02	 2.5284981e-01	[-4.4523614e-02]	 1.9624782e-01
       8	-3.4031103e-02	 2.4531080e-01	-1.3534105e-02	 2.4825355e-01	[-2.6026690e-02]	 1.9897652e-01


.. parsed-literal::

       9	-2.3604438e-02	 2.4328471e-01	-5.4860894e-03	 2.4595230e-01	[-1.7509610e-02]	 1.9410944e-01


.. parsed-literal::

      10	-1.3851017e-02	 2.4171596e-01	 2.2474691e-03	 2.4437429e-01	[-9.0870753e-03]	 2.1043396e-01
      11	-8.2962516e-03	 2.4059951e-01	 7.1653935e-03	 2.4383376e-01	[-6.6843676e-03]	 1.8697882e-01


.. parsed-literal::

      12	-5.4660359e-03	 2.4012232e-01	 9.5390060e-03	 2.4347790e-01	[-5.1783677e-03]	 2.0361876e-01
      13	 1.0606857e-04	 2.3908322e-01	 1.5194357e-02	 2.4239355e-01	[ 9.7732308e-04]	 1.9769478e-01


.. parsed-literal::

      14	 1.0002636e-01	 2.2620150e-01	 1.2286780e-01	 2.2861258e-01	[ 1.0751264e-01]	 3.0907655e-01


.. parsed-literal::

      15	 1.2634505e-01	 2.2491787e-01	 1.4871852e-01	 2.2703596e-01	[ 1.3515632e-01]	 3.1806397e-01


.. parsed-literal::

      16	 1.8608191e-01	 2.2119221e-01	 2.1112292e-01	 2.2319162e-01	[ 1.9455273e-01]	 2.1258402e-01


.. parsed-literal::

      17	 2.6642997e-01	 2.1786645e-01	 2.9970315e-01	 2.2021057e-01	[ 2.7242783e-01]	 2.0911384e-01


.. parsed-literal::

      18	 3.0658724e-01	 2.1382571e-01	 3.3857730e-01	 2.1914194e-01	[ 3.0034363e-01]	 2.0969129e-01
      19	 3.4859474e-01	 2.1101553e-01	 3.8054547e-01	 2.1623270e-01	[ 3.4508026e-01]	 2.0319057e-01


.. parsed-literal::

      20	 4.1133357e-01	 2.0693775e-01	 4.4452120e-01	 2.1256602e-01	[ 4.0321356e-01]	 1.9667125e-01
      21	 5.9413123e-01	 2.0730924e-01	 6.3373125e-01	 2.1320280e-01	[ 5.4517193e-01]	 1.9936347e-01


.. parsed-literal::

      22	 6.2651031e-01	 2.0545710e-01	 6.6761913e-01	 2.0936556e-01	[ 5.8262093e-01]	 2.9416728e-01
      23	 6.6247835e-01	 2.0126250e-01	 7.0148986e-01	 2.0592763e-01	[ 6.2806643e-01]	 1.7847562e-01


.. parsed-literal::

      24	 6.8128288e-01	 2.0163477e-01	 7.2025333e-01	 2.0531832e-01	[ 6.4443320e-01]	 1.7555475e-01
      25	 7.1395033e-01	 2.0027649e-01	 7.5073724e-01	 2.0304883e-01	[ 6.8271636e-01]	 2.0173216e-01


.. parsed-literal::

      26	 7.4607675e-01	 1.9843233e-01	 7.8472355e-01	 2.0021742e-01	[ 7.1256667e-01]	 2.1016598e-01
      27	 7.6837374e-01	 1.9647746e-01	 8.0825659e-01	 1.9721106e-01	[ 7.2997443e-01]	 1.8210459e-01


.. parsed-literal::

      28	 7.9102655e-01	 1.9351861e-01	 8.3143880e-01	 1.9417123e-01	[ 7.5279367e-01]	 1.8038845e-01
      29	 8.2077244e-01	 1.8916036e-01	 8.6156410e-01	 1.9046076e-01	[ 7.8344403e-01]	 1.8304896e-01


.. parsed-literal::

      30	 8.5027648e-01	 1.8513657e-01	 8.9177344e-01	 1.8605277e-01	[ 8.2592014e-01]	 2.0253420e-01
      31	 8.6977681e-01	 1.8273000e-01	 9.1166778e-01	 1.8513105e-01	[ 8.4042849e-01]	 1.9347358e-01


.. parsed-literal::

      32	 8.8675149e-01	 1.8099554e-01	 9.2956394e-01	 1.8406137e-01	[ 8.5110365e-01]	 1.9646835e-01


.. parsed-literal::

      33	 9.0861411e-01	 1.7946015e-01	 9.5173649e-01	 1.8201856e-01	[ 8.7774574e-01]	 2.0551634e-01
      34	 9.3559629e-01	 1.7776134e-01	 9.7989498e-01	 1.7867389e-01	[ 9.0287031e-01]	 1.9137120e-01


.. parsed-literal::

      35	 9.5198863e-01	 1.8037167e-01	 9.9738227e-01	 1.7950981e-01	[ 9.1626240e-01]	 2.1404934e-01
      36	 9.6999039e-01	 1.7757496e-01	 1.0158023e+00	 1.7759344e-01	[ 9.3248281e-01]	 1.7966533e-01


.. parsed-literal::

      37	 9.8286415e-01	 1.7547682e-01	 1.0286822e+00	 1.7644816e-01	[ 9.4691097e-01]	 1.6632342e-01


.. parsed-literal::

      38	 1.0026819e+00	 1.7462733e-01	 1.0493622e+00	 1.7621883e-01	[ 9.6857275e-01]	 2.0309925e-01
      39	 1.0046880e+00	 1.7432542e-01	 1.0526841e+00	 1.7461557e-01	[ 9.7177405e-01]	 1.9751453e-01


.. parsed-literal::

      40	 1.0201949e+00	 1.7377427e-01	 1.0676836e+00	 1.7468540e-01	[ 9.8546101e-01]	 1.7030478e-01
      41	 1.0259222e+00	 1.7335173e-01	 1.0734582e+00	 1.7428060e-01	[ 9.8921970e-01]	 1.9320607e-01


.. parsed-literal::

      42	 1.0324893e+00	 1.7256819e-01	 1.0801860e+00	 1.7326632e-01	[ 9.9398815e-01]	 1.8406534e-01


.. parsed-literal::

      43	 1.0437554e+00	 1.7097432e-01	 1.0917126e+00	 1.7140421e-01	[ 1.0039194e+00]	 2.0258403e-01


.. parsed-literal::

      44	 1.0532753e+00	 1.7056351e-01	 1.1019777e+00	 1.6902163e-01	[ 1.0206588e+00]	 2.0330763e-01


.. parsed-literal::

      45	 1.0649235e+00	 1.6858401e-01	 1.1133982e+00	 1.6743660e-01	[ 1.0307695e+00]	 2.0761061e-01
      46	 1.0701348e+00	 1.6789721e-01	 1.1182851e+00	 1.6726346e-01	[ 1.0347298e+00]	 1.7076492e-01


.. parsed-literal::

      47	 1.0796064e+00	 1.6692359e-01	 1.1277111e+00	 1.6633228e-01	[ 1.0460410e+00]	 2.0432043e-01
      48	 1.0863023e+00	 1.6617255e-01	 1.1351514e+00	 1.6467329e-01	[ 1.0521325e+00]	 1.9197679e-01


.. parsed-literal::

      49	 1.0936842e+00	 1.6542385e-01	 1.1425007e+00	 1.6397450e-01	[ 1.0621326e+00]	 1.9145370e-01
      50	 1.0990800e+00	 1.6476524e-01	 1.1479159e+00	 1.6339344e-01	[ 1.0676220e+00]	 2.0020747e-01


.. parsed-literal::

      51	 1.1068062e+00	 1.6395880e-01	 1.1557653e+00	 1.6254261e-01	[ 1.0746009e+00]	 1.8953967e-01


.. parsed-literal::

      52	 1.1178500e+00	 1.6333553e-01	 1.1672060e+00	 1.6176321e-01	[ 1.0814584e+00]	 2.0761943e-01
      53	 1.1237460e+00	 1.6209029e-01	 1.1735538e+00	 1.6053675e-01	[ 1.0827246e+00]	 2.0014191e-01


.. parsed-literal::

      54	 1.1293660e+00	 1.6204914e-01	 1.1789897e+00	 1.6045355e-01	[ 1.0883913e+00]	 1.9082928e-01
      55	 1.1345768e+00	 1.6175481e-01	 1.1841994e+00	 1.6020154e-01	[ 1.0917113e+00]	 1.7786479e-01


.. parsed-literal::

      56	 1.1415502e+00	 1.6104463e-01	 1.1914057e+00	 1.5952499e-01	[ 1.0964756e+00]	 1.9555545e-01
      57	 1.1443816e+00	 1.6039970e-01	 1.1948582e+00	 1.5892739e-01	  1.0959596e+00 	 1.7683864e-01


.. parsed-literal::

      58	 1.1530432e+00	 1.5969412e-01	 1.2033248e+00	 1.5844472e-01	[ 1.1053650e+00]	 1.9709325e-01
      59	 1.1561036e+00	 1.5933645e-01	 1.2063513e+00	 1.5815905e-01	[ 1.1078916e+00]	 1.9428349e-01


.. parsed-literal::

      60	 1.1632628e+00	 1.5872973e-01	 1.2138402e+00	 1.5731432e-01	[ 1.1128281e+00]	 2.1252680e-01


.. parsed-literal::

      61	 1.1703725e+00	 1.5770538e-01	 1.2214560e+00	 1.5683447e-01	  1.1076545e+00 	 2.0836306e-01
      62	 1.1782347e+00	 1.5755738e-01	 1.2295734e+00	 1.5626346e-01	[ 1.1186216e+00]	 1.9207883e-01


.. parsed-literal::

      63	 1.1851203e+00	 1.5731636e-01	 1.2365526e+00	 1.5582513e-01	[ 1.1260483e+00]	 2.0548344e-01
      64	 1.1951704e+00	 1.5708177e-01	 1.2468949e+00	 1.5528655e-01	[ 1.1352845e+00]	 1.9764709e-01


.. parsed-literal::

      65	 1.1977170e+00	 1.5722283e-01	 1.2500137e+00	 1.5542837e-01	  1.1314534e+00 	 1.9967842e-01


.. parsed-literal::

      66	 1.2058014e+00	 1.5645128e-01	 1.2576905e+00	 1.5490544e-01	[ 1.1415605e+00]	 2.0420384e-01


.. parsed-literal::

      67	 1.2095039e+00	 1.5594131e-01	 1.2614047e+00	 1.5459082e-01	[ 1.1451753e+00]	 2.0446301e-01
      68	 1.2146773e+00	 1.5526952e-01	 1.2666910e+00	 1.5423865e-01	[ 1.1491003e+00]	 1.9942021e-01


.. parsed-literal::

      69	 1.2209409e+00	 1.5443912e-01	 1.2732481e+00	 1.5382922e-01	[ 1.1519827e+00]	 1.9904685e-01
      70	 1.2270407e+00	 1.5370210e-01	 1.2798473e+00	 1.5366434e-01	[ 1.1522665e+00]	 1.9364142e-01


.. parsed-literal::

      71	 1.2316380e+00	 1.5344810e-01	 1.2843863e+00	 1.5335831e-01	[ 1.1553781e+00]	 1.9938588e-01


.. parsed-literal::

      72	 1.2366975e+00	 1.5334672e-01	 1.2895756e+00	 1.5313610e-01	[ 1.1573612e+00]	 2.1091008e-01


.. parsed-literal::

      73	 1.2421363e+00	 1.5300990e-01	 1.2952678e+00	 1.5284426e-01	  1.1512835e+00 	 2.1212149e-01
      74	 1.2481091e+00	 1.5289558e-01	 1.3013869e+00	 1.5293184e-01	  1.1509102e+00 	 1.8199253e-01


.. parsed-literal::

      75	 1.2529505e+00	 1.5243326e-01	 1.3062692e+00	 1.5283818e-01	  1.1525994e+00 	 2.0046401e-01


.. parsed-literal::

      76	 1.2594055e+00	 1.5177717e-01	 1.3128263e+00	 1.5246497e-01	  1.1534942e+00 	 2.0608592e-01
      77	 1.2633346e+00	 1.5147162e-01	 1.3170752e+00	 1.5258201e-01	  1.1460966e+00 	 1.7741919e-01


.. parsed-literal::

      78	 1.2698062e+00	 1.5119608e-01	 1.3234091e+00	 1.5215215e-01	  1.1532159e+00 	 1.8755007e-01


.. parsed-literal::

      79	 1.2737381e+00	 1.5106742e-01	 1.3273741e+00	 1.5187625e-01	  1.1523378e+00 	 2.0158839e-01
      80	 1.2783352e+00	 1.5085259e-01	 1.3319592e+00	 1.5160317e-01	  1.1541566e+00 	 1.7250967e-01


.. parsed-literal::

      81	 1.2843068e+00	 1.5026334e-01	 1.3381844e+00	 1.5163452e-01	  1.1460214e+00 	 2.1058249e-01
      82	 1.2890895e+00	 1.4989597e-01	 1.3430321e+00	 1.5151574e-01	  1.1534437e+00 	 1.7452955e-01


.. parsed-literal::

      83	 1.2924414e+00	 1.4968061e-01	 1.3462837e+00	 1.5149409e-01	[ 1.1583506e+00]	 1.9531345e-01


.. parsed-literal::

      84	 1.2971585e+00	 1.4944266e-01	 1.3510655e+00	 1.5144593e-01	[ 1.1612203e+00]	 2.0746017e-01


.. parsed-literal::

      85	 1.3025693e+00	 1.4911252e-01	 1.3566508e+00	 1.5104854e-01	[ 1.1629618e+00]	 2.0309114e-01
      86	 1.3061230e+00	 1.4923939e-01	 1.3607418e+00	 1.5008573e-01	  1.1624877e+00 	 1.9409704e-01


.. parsed-literal::

      87	 1.3135924e+00	 1.4892790e-01	 1.3680116e+00	 1.5011262e-01	[ 1.1695226e+00]	 1.7154694e-01
      88	 1.3161826e+00	 1.4867444e-01	 1.3705819e+00	 1.5000142e-01	[ 1.1695909e+00]	 1.8743968e-01


.. parsed-literal::

      89	 1.3214316e+00	 1.4821647e-01	 1.3760747e+00	 1.4982960e-01	  1.1678165e+00 	 2.0113087e-01


.. parsed-literal::

      90	 1.3246147e+00	 1.4760049e-01	 1.3794607e+00	 1.4941192e-01	  1.1613085e+00 	 2.9969549e-01
      91	 1.3284776e+00	 1.4731808e-01	 1.3834938e+00	 1.4937584e-01	  1.1585758e+00 	 1.9828200e-01


.. parsed-literal::

      92	 1.3337483e+00	 1.4684588e-01	 1.3889635e+00	 1.4898531e-01	  1.1538694e+00 	 2.0510459e-01


.. parsed-literal::

      93	 1.3367079e+00	 1.4615316e-01	 1.3921234e+00	 1.4846399e-01	  1.1482876e+00 	 2.0337367e-01
      94	 1.3406040e+00	 1.4588367e-01	 1.3959545e+00	 1.4786853e-01	  1.1487717e+00 	 1.8102741e-01


.. parsed-literal::

      95	 1.3449660e+00	 1.4535124e-01	 1.4002991e+00	 1.4707768e-01	  1.1480510e+00 	 1.6050625e-01
      96	 1.3495221e+00	 1.4459269e-01	 1.4048745e+00	 1.4636915e-01	  1.1473146e+00 	 1.7282534e-01


.. parsed-literal::

      97	 1.3519963e+00	 1.4398121e-01	 1.4074569e+00	 1.4586983e-01	  1.1472186e+00 	 3.0699778e-01


.. parsed-literal::

      98	 1.3557729e+00	 1.4323326e-01	 1.4112981e+00	 1.4580328e-01	  1.1414206e+00 	 2.1252441e-01
      99	 1.3584344e+00	 1.4291372e-01	 1.4139756e+00	 1.4605666e-01	  1.1412078e+00 	 1.9769025e-01


.. parsed-literal::

     100	 1.3618860e+00	 1.4243931e-01	 1.4175234e+00	 1.4628891e-01	  1.1362963e+00 	 1.8250036e-01
     101	 1.3659519e+00	 1.4167828e-01	 1.4216481e+00	 1.4575445e-01	  1.1403777e+00 	 1.6848063e-01


.. parsed-literal::

     102	 1.3703187e+00	 1.4067360e-01	 1.4260852e+00	 1.4513974e-01	  1.1375716e+00 	 2.0430779e-01


.. parsed-literal::

     103	 1.3739454e+00	 1.4002440e-01	 1.4296688e+00	 1.4432870e-01	  1.1397569e+00 	 2.0375514e-01
     104	 1.3773098e+00	 1.3967952e-01	 1.4329542e+00	 1.4352848e-01	  1.1468104e+00 	 1.8035531e-01


.. parsed-literal::

     105	 1.3806936e+00	 1.3936188e-01	 1.4364036e+00	 1.4336466e-01	  1.1497118e+00 	 1.6815519e-01
     106	 1.3845979e+00	 1.3881653e-01	 1.4404649e+00	 1.4339625e-01	  1.1519566e+00 	 1.9443464e-01


.. parsed-literal::

     107	 1.3875429e+00	 1.3878762e-01	 1.4435261e+00	 1.4369379e-01	  1.1527389e+00 	 1.8703675e-01
     108	 1.3907609e+00	 1.3862908e-01	 1.4467761e+00	 1.4386856e-01	  1.1515668e+00 	 1.8210769e-01


.. parsed-literal::

     109	 1.3950857e+00	 1.3814582e-01	 1.4513410e+00	 1.4389428e-01	  1.1454340e+00 	 1.9574904e-01
     110	 1.3976737e+00	 1.3766793e-01	 1.4540776e+00	 1.4337774e-01	  1.1377013e+00 	 1.9510627e-01


.. parsed-literal::

     111	 1.4001786e+00	 1.3758665e-01	 1.4564437e+00	 1.4300987e-01	  1.1437226e+00 	 1.7203140e-01
     112	 1.4029806e+00	 1.3726122e-01	 1.4592789e+00	 1.4244343e-01	  1.1452989e+00 	 1.7469430e-01


.. parsed-literal::

     113	 1.4057474e+00	 1.3698582e-01	 1.4621350e+00	 1.4199640e-01	  1.1463879e+00 	 2.1428633e-01
     114	 1.4100432e+00	 1.3634568e-01	 1.4666214e+00	 1.4141368e-01	  1.1426603e+00 	 1.9859648e-01


.. parsed-literal::

     115	 1.4123376e+00	 1.3595198e-01	 1.4691174e+00	 1.4105626e-01	  1.1452062e+00 	 1.8032050e-01
     116	 1.4154880e+00	 1.3599105e-01	 1.4720145e+00	 1.4123450e-01	  1.1477894e+00 	 2.0413494e-01


.. parsed-literal::

     117	 1.4177908e+00	 1.3582843e-01	 1.4742481e+00	 1.4129951e-01	  1.1472507e+00 	 2.0665884e-01
     118	 1.4208322e+00	 1.3557262e-01	 1.4773019e+00	 1.4130509e-01	  1.1449467e+00 	 1.8027306e-01


.. parsed-literal::

     119	 1.4242508e+00	 1.3517347e-01	 1.4808053e+00	 1.4098009e-01	  1.1459205e+00 	 1.9412184e-01
     120	 1.4271525e+00	 1.3495946e-01	 1.4838062e+00	 1.4060446e-01	  1.1468395e+00 	 1.7627048e-01


.. parsed-literal::

     121	 1.4300727e+00	 1.3482249e-01	 1.4867962e+00	 1.4023980e-01	  1.1493428e+00 	 2.0450878e-01


.. parsed-literal::

     122	 1.4327669e+00	 1.3483014e-01	 1.4895505e+00	 1.3973902e-01	  1.1539799e+00 	 2.1367836e-01
     123	 1.4355842e+00	 1.3477082e-01	 1.4923785e+00	 1.3965188e-01	  1.1572502e+00 	 2.0066500e-01


.. parsed-literal::

     124	 1.4378145e+00	 1.3463341e-01	 1.4945736e+00	 1.3957734e-01	  1.1637155e+00 	 1.7300415e-01


.. parsed-literal::

     125	 1.4395905e+00	 1.3436905e-01	 1.4964362e+00	 1.3974863e-01	[ 1.1710838e+00]	 2.0682931e-01


.. parsed-literal::

     126	 1.4415731e+00	 1.3422846e-01	 1.4983195e+00	 1.3964454e-01	[ 1.1767399e+00]	 2.0856261e-01


.. parsed-literal::

     127	 1.4431709e+00	 1.3404530e-01	 1.4999234e+00	 1.3954312e-01	[ 1.1808332e+00]	 2.0155382e-01
     128	 1.4451809e+00	 1.3382988e-01	 1.5019885e+00	 1.3948807e-01	[ 1.1852130e+00]	 1.6646433e-01


.. parsed-literal::

     129	 1.4479343e+00	 1.3360895e-01	 1.5048377e+00	 1.3938627e-01	[ 1.1891637e+00]	 2.0074987e-01


.. parsed-literal::

     130	 1.4496082e+00	 1.3354528e-01	 1.5065398e+00	 1.3927171e-01	[ 1.1918089e+00]	 2.9084921e-01


.. parsed-literal::

     131	 1.4518528e+00	 1.3344390e-01	 1.5088039e+00	 1.3905984e-01	[ 1.1933187e+00]	 2.0472288e-01
     132	 1.4536295e+00	 1.3335256e-01	 1.5105753e+00	 1.3873730e-01	  1.1923244e+00 	 1.8075347e-01


.. parsed-literal::

     133	 1.4553223e+00	 1.3335191e-01	 1.5122363e+00	 1.3844962e-01	  1.1916313e+00 	 2.0557928e-01


.. parsed-literal::

     134	 1.4572530e+00	 1.3330495e-01	 1.5141124e+00	 1.3820061e-01	  1.1902320e+00 	 2.1441412e-01


.. parsed-literal::

     135	 1.4603257e+00	 1.3321736e-01	 1.5171495e+00	 1.3782869e-01	  1.1865314e+00 	 2.1600080e-01
     136	 1.4622966e+00	 1.3316650e-01	 1.5191815e+00	 1.3773271e-01	  1.1783271e+00 	 1.9757223e-01


.. parsed-literal::

     137	 1.4646639e+00	 1.3303950e-01	 1.5215580e+00	 1.3758836e-01	  1.1772452e+00 	 1.6941166e-01
     138	 1.4669086e+00	 1.3287593e-01	 1.5238966e+00	 1.3754658e-01	  1.1755645e+00 	 1.8612075e-01


.. parsed-literal::

     139	 1.4690128e+00	 1.3271348e-01	 1.5261245e+00	 1.3764797e-01	  1.1710248e+00 	 1.9474077e-01
     140	 1.4712239e+00	 1.3250457e-01	 1.5285359e+00	 1.3764620e-01	  1.1662884e+00 	 1.9973207e-01


.. parsed-literal::

     141	 1.4732454e+00	 1.3242113e-01	 1.5305604e+00	 1.3778452e-01	  1.1640186e+00 	 2.0438671e-01


.. parsed-literal::

     142	 1.4756762e+00	 1.3235694e-01	 1.5329485e+00	 1.3798408e-01	  1.1634064e+00 	 2.0168161e-01


.. parsed-literal::

     143	 1.4775099e+00	 1.3229851e-01	 1.5347788e+00	 1.3808254e-01	  1.1640946e+00 	 2.0838356e-01
     144	 1.4793724e+00	 1.3222883e-01	 1.5366163e+00	 1.3805881e-01	  1.1659258e+00 	 1.8856502e-01


.. parsed-literal::

     145	 1.4812728e+00	 1.3202394e-01	 1.5386546e+00	 1.3792384e-01	  1.1664118e+00 	 1.7836976e-01
     146	 1.4833380e+00	 1.3178350e-01	 1.5407705e+00	 1.3771088e-01	  1.1629066e+00 	 1.8886852e-01


.. parsed-literal::

     147	 1.4846280e+00	 1.3165657e-01	 1.5420741e+00	 1.3760470e-01	  1.1611232e+00 	 1.8499136e-01
     148	 1.4870556e+00	 1.3123524e-01	 1.5447205e+00	 1.3736372e-01	  1.1507230e+00 	 1.9320512e-01


.. parsed-literal::

     149	 1.4874940e+00	 1.3104549e-01	 1.5453456e+00	 1.3732901e-01	  1.1459493e+00 	 2.0512915e-01
     150	 1.4891987e+00	 1.3107925e-01	 1.5468687e+00	 1.3729811e-01	  1.1505596e+00 	 1.7199945e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min, sys: 952 ms, total: 2min 1s
    Wall time: 30.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f5e3ce7f0d0>



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
    CPU times: user 948 ms, sys: 57 ms, total: 1.01 s
    Wall time: 375 ms


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

