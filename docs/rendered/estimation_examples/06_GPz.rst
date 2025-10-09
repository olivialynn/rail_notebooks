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

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.5458463e-01	 3.2388913e-01	-3.4501681e-01	 3.0743912e-01	[-3.1511721e-01]	 4.6425748e-01


.. parsed-literal::

       2	-2.8571372e-01	 3.1449164e-01	-2.6325732e-01	 2.9710303e-01	[-2.1224641e-01]	 2.3407435e-01


.. parsed-literal::

       3	-2.4063267e-01	 2.9259227e-01	-1.9874887e-01	 2.7242773e-01	[-1.1686278e-01]	 3.0333424e-01
       4	-1.9465640e-01	 2.6900080e-01	-1.5201289e-01	 2.5005365e-01	[-4.9550956e-02]	 1.6601396e-01


.. parsed-literal::

       5	-1.0634546e-01	 2.5840869e-01	-7.5966420e-02	 2.4388746e-01	[-6.6830537e-03]	 1.9595218e-01
       6	-7.6654586e-02	 2.5373918e-01	-4.8827313e-02	 2.4073620e-01	[-2.7064446e-03]	 1.9877720e-01


.. parsed-literal::

       7	-5.8397263e-02	 2.5065197e-01	-3.5616619e-02	 2.3937736e-01	[ 7.3500978e-03]	 2.1469188e-01


.. parsed-literal::

       8	-4.5232357e-02	 2.4836475e-01	-2.5991763e-02	 2.3828548e-01	[ 1.5106007e-02]	 2.1334600e-01


.. parsed-literal::

       9	-3.2490285e-02	 2.4582028e-01	-1.5331462e-02	 2.3664665e-01	[ 2.3452868e-02]	 2.1166587e-01


.. parsed-literal::

      10	-2.6096596e-02	 2.4406959e-01	-1.1249382e-02	 2.3677132e-01	[ 2.6659427e-02]	 2.0930505e-01


.. parsed-literal::

      11	-1.8385876e-02	 2.4316700e-01	-3.8038087e-03	 2.3678161e-01	  2.5990820e-02 	 2.1344018e-01


.. parsed-literal::

      12	-1.6168662e-02	 2.4273723e-01	-1.7896688e-03	 2.3633989e-01	[ 2.7979233e-02]	 2.1967721e-01


.. parsed-literal::

      13	-1.2453411e-02	 2.4204329e-01	 1.7242958e-03	 2.3560673e-01	[ 3.1241300e-02]	 2.1361256e-01


.. parsed-literal::

      14	 1.2899567e-01	 2.2290407e-01	 1.5205963e-01	 2.1543363e-01	[ 1.7458350e-01]	 4.4192338e-01


.. parsed-literal::

      15	 1.8594384e-01	 2.1967274e-01	 2.1066554e-01	 2.1180969e-01	[ 2.3049030e-01]	 2.0712495e-01


.. parsed-literal::

      16	 2.8859269e-01	 2.1532036e-01	 3.1714871e-01	 2.0969099e-01	[ 3.4077231e-01]	 2.1528864e-01


.. parsed-literal::

      17	 3.4119864e-01	 2.1139788e-01	 3.7313480e-01	 2.0704752e-01	[ 4.1208171e-01]	 2.2209883e-01


.. parsed-literal::

      18	 3.9889277e-01	 2.0909598e-01	 4.3218672e-01	 2.0369670e-01	[ 4.7704640e-01]	 2.1159005e-01


.. parsed-literal::

      19	 4.5010105e-01	 2.0623027e-01	 4.8340050e-01	 1.9807247e-01	[ 5.2369560e-01]	 2.2129321e-01


.. parsed-literal::

      20	 5.3595381e-01	 2.0107666e-01	 5.7036329e-01	 1.9191392e-01	[ 5.9796723e-01]	 2.1269107e-01


.. parsed-literal::

      21	 6.2485466e-01	 1.9764240e-01	 6.6309652e-01	 1.8994828e-01	[ 6.6014771e-01]	 2.1029210e-01


.. parsed-literal::

      22	 6.6884611e-01	 1.9144725e-01	 7.0853789e-01	 1.8332613e-01	[ 6.7809180e-01]	 2.1586657e-01


.. parsed-literal::

      23	 7.0636866e-01	 1.8943079e-01	 7.4354984e-01	 1.8245330e-01	[ 7.2021121e-01]	 2.1392345e-01


.. parsed-literal::

      24	 7.3301476e-01	 1.8864428e-01	 7.7044162e-01	 1.8226331e-01	[ 7.4203261e-01]	 2.1940565e-01


.. parsed-literal::

      25	 7.4405944e-01	 1.9624437e-01	 7.8074871e-01	 1.9476662e-01	[ 7.4605869e-01]	 2.1088004e-01


.. parsed-literal::

      26	 8.0131156e-01	 1.9297479e-01	 8.3862331e-01	 1.8940771e-01	[ 8.0978911e-01]	 2.0542336e-01
      27	 8.2824435e-01	 1.8644120e-01	 8.6669333e-01	 1.8320890e-01	[ 8.3441092e-01]	 2.1291304e-01


.. parsed-literal::

      28	 8.5492928e-01	 1.8307934e-01	 8.9449364e-01	 1.7923970e-01	[ 8.6031182e-01]	 2.1198630e-01


.. parsed-literal::

      29	 8.9329919e-01	 1.7958811e-01	 9.3475567e-01	 1.7461936e-01	[ 8.9311218e-01]	 2.0875502e-01


.. parsed-literal::

      30	 9.1056882e-01	 1.7831072e-01	 9.5289236e-01	 1.7220019e-01	[ 9.0133483e-01]	 2.0119739e-01


.. parsed-literal::

      31	 9.3167512e-01	 1.7568755e-01	 9.7331769e-01	 1.7172839e-01	[ 9.1861665e-01]	 2.1519780e-01


.. parsed-literal::

      32	 9.4179243e-01	 1.7407672e-01	 9.8403925e-01	 1.7131828e-01	[ 9.2720179e-01]	 2.0755529e-01


.. parsed-literal::

      33	 9.5693911e-01	 1.7164842e-01	 1.0000654e+00	 1.6996139e-01	[ 9.3962821e-01]	 2.1174145e-01


.. parsed-literal::

      34	 9.7623063e-01	 1.6919361e-01	 1.0204743e+00	 1.6821062e-01	[ 9.5319318e-01]	 2.0398307e-01


.. parsed-literal::

      35	 9.9123106e-01	 1.6708649e-01	 1.0361287e+00	 1.6601118e-01	[ 9.6631084e-01]	 2.1120071e-01


.. parsed-literal::

      36	 1.0066885e+00	 1.6576568e-01	 1.0512639e+00	 1.6612337e-01	[ 9.7890540e-01]	 2.1334028e-01


.. parsed-literal::

      37	 1.0216237e+00	 1.6391719e-01	 1.0663967e+00	 1.6591954e-01	[ 9.9290536e-01]	 2.0324421e-01


.. parsed-literal::

      38	 1.0346176e+00	 1.6283406e-01	 1.0797649e+00	 1.6540360e-01	[ 1.0018740e+00]	 2.2003460e-01


.. parsed-literal::

      39	 1.0466557e+00	 1.6157434e-01	 1.0921960e+00	 1.6362836e-01	[ 1.0153000e+00]	 2.0985723e-01


.. parsed-literal::

      40	 1.0558008e+00	 1.6059914e-01	 1.1018002e+00	 1.6140929e-01	[ 1.0260577e+00]	 2.0865488e-01
      41	 1.0731766e+00	 1.5891243e-01	 1.1197719e+00	 1.5754996e-01	[ 1.0400210e+00]	 1.9001818e-01


.. parsed-literal::

      42	 1.0835354e+00	 1.5746446e-01	 1.1306703e+00	 1.5344524e-01	[ 1.0638782e+00]	 2.0854568e-01


.. parsed-literal::

      43	 1.0969735e+00	 1.5633210e-01	 1.1437850e+00	 1.5329878e-01	[ 1.0643413e+00]	 2.0665646e-01


.. parsed-literal::

      44	 1.1034979e+00	 1.5557257e-01	 1.1499948e+00	 1.5333497e-01	[ 1.0646728e+00]	 2.1126652e-01
      45	 1.1144674e+00	 1.5478044e-01	 1.1613568e+00	 1.5348787e-01	[ 1.0653280e+00]	 1.7837405e-01


.. parsed-literal::

      46	 1.1266648e+00	 1.5358746e-01	 1.1739620e+00	 1.5251458e-01	[ 1.0691141e+00]	 2.1768832e-01
      47	 1.1363875e+00	 1.5269403e-01	 1.1839167e+00	 1.5144986e-01	[ 1.0814587e+00]	 1.9463849e-01


.. parsed-literal::

      48	 1.1459702e+00	 1.5203398e-01	 1.1938061e+00	 1.5088212e-01	[ 1.0918414e+00]	 2.1253991e-01
      49	 1.1578117e+00	 1.5098039e-01	 1.2064051e+00	 1.4998880e-01	[ 1.0984522e+00]	 1.8187904e-01


.. parsed-literal::

      50	 1.1654070e+00	 1.5070440e-01	 1.2140556e+00	 1.5124655e-01	[ 1.1133684e+00]	 2.0323706e-01
      51	 1.1712700e+00	 1.5038077e-01	 1.2196902e+00	 1.5116103e-01	[ 1.1168465e+00]	 1.9941926e-01


.. parsed-literal::

      52	 1.1794531e+00	 1.4983334e-01	 1.2279283e+00	 1.5141067e-01	[ 1.1199994e+00]	 2.1117520e-01
      53	 1.1863290e+00	 1.4999841e-01	 1.2349961e+00	 1.5256272e-01	[ 1.1255292e+00]	 1.7646861e-01


.. parsed-literal::

      54	 1.1950778e+00	 1.5048560e-01	 1.2442125e+00	 1.5485144e-01	[ 1.1313444e+00]	 2.1564388e-01


.. parsed-literal::

      55	 1.2038066e+00	 1.5088098e-01	 1.2528962e+00	 1.5557894e-01	[ 1.1427282e+00]	 2.0874953e-01
      56	 1.2082927e+00	 1.5004527e-01	 1.2572380e+00	 1.5439337e-01	[ 1.1502121e+00]	 1.8114495e-01


.. parsed-literal::

      57	 1.2182916e+00	 1.4877778e-01	 1.2677916e+00	 1.5401427e-01	  1.1501472e+00 	 2.1711016e-01


.. parsed-literal::

      58	 1.2268142e+00	 1.4826968e-01	 1.2762930e+00	 1.5342336e-01	[ 1.1589042e+00]	 2.1505976e-01


.. parsed-literal::

      59	 1.2338873e+00	 1.4804410e-01	 1.2832550e+00	 1.5323623e-01	[ 1.1683051e+00]	 2.1607447e-01


.. parsed-literal::

      60	 1.2444282e+00	 1.4764455e-01	 1.2939567e+00	 1.5365058e-01	[ 1.1760524e+00]	 2.1105862e-01


.. parsed-literal::

      61	 1.2497482e+00	 1.4735145e-01	 1.2995384e+00	 1.5272490e-01	[ 1.1887323e+00]	 2.0529175e-01


.. parsed-literal::

      62	 1.2557363e+00	 1.4680100e-01	 1.3054008e+00	 1.5242188e-01	[ 1.1925276e+00]	 2.1704412e-01


.. parsed-literal::

      63	 1.2633634e+00	 1.4592168e-01	 1.3134464e+00	 1.5254630e-01	  1.1893639e+00 	 2.1040392e-01


.. parsed-literal::

      64	 1.2686332e+00	 1.4526598e-01	 1.3188874e+00	 1.5167773e-01	[ 1.1947447e+00]	 2.1144247e-01


.. parsed-literal::

      65	 1.2767713e+00	 1.4419276e-01	 1.3271843e+00	 1.5051093e-01	[ 1.2000000e+00]	 2.1116996e-01


.. parsed-literal::

      66	 1.2855270e+00	 1.4311287e-01	 1.3363716e+00	 1.4894252e-01	[ 1.2005050e+00]	 2.0576596e-01
      67	 1.2934198e+00	 1.4263627e-01	 1.3445124e+00	 1.4733524e-01	[ 1.2063851e+00]	 1.7457128e-01


.. parsed-literal::

      68	 1.3003473e+00	 1.4223190e-01	 1.3516243e+00	 1.4660046e-01	[ 1.2078015e+00]	 1.9312644e-01


.. parsed-literal::

      69	 1.3060259e+00	 1.4215522e-01	 1.3572833e+00	 1.4539938e-01	[ 1.2178549e+00]	 2.1286798e-01


.. parsed-literal::

      70	 1.3106621e+00	 1.4162456e-01	 1.3619677e+00	 1.4520645e-01	[ 1.2182508e+00]	 2.1376753e-01


.. parsed-literal::

      71	 1.3194292e+00	 1.4030222e-01	 1.3710096e+00	 1.4431199e-01	  1.2165050e+00 	 2.0444942e-01


.. parsed-literal::

      72	 1.3253054e+00	 1.3983743e-01	 1.3767471e+00	 1.4337513e-01	[ 1.2220623e+00]	 2.1999359e-01


.. parsed-literal::

      73	 1.3316236e+00	 1.3934298e-01	 1.3829718e+00	 1.4231452e-01	[ 1.2292579e+00]	 2.1435404e-01


.. parsed-literal::

      74	 1.3387948e+00	 1.3851690e-01	 1.3903516e+00	 1.4047246e-01	[ 1.2315986e+00]	 2.1381736e-01


.. parsed-literal::

      75	 1.3441882e+00	 1.3819450e-01	 1.3958563e+00	 1.3985828e-01	[ 1.2336218e+00]	 2.0384383e-01


.. parsed-literal::

      76	 1.3511616e+00	 1.3770298e-01	 1.4028357e+00	 1.3919023e-01	[ 1.2365309e+00]	 2.1052790e-01
      77	 1.3585850e+00	 1.3663669e-01	 1.4107149e+00	 1.3771622e-01	  1.2288112e+00 	 1.9690299e-01


.. parsed-literal::

      78	 1.3652258e+00	 1.3635336e-01	 1.4172888e+00	 1.3748286e-01	  1.2360830e+00 	 2.1483397e-01


.. parsed-literal::

      79	 1.3694795e+00	 1.3564966e-01	 1.4214796e+00	 1.3701636e-01	[ 1.2410610e+00]	 2.1532750e-01


.. parsed-literal::

      80	 1.3745460e+00	 1.3454814e-01	 1.4266785e+00	 1.3580337e-01	[ 1.2418173e+00]	 2.1514106e-01


.. parsed-literal::

      81	 1.3785484e+00	 1.3346675e-01	 1.4309535e+00	 1.3423905e-01	[ 1.2464016e+00]	 2.1246266e-01


.. parsed-literal::

      82	 1.3832762e+00	 1.3298614e-01	 1.4355777e+00	 1.3387075e-01	  1.2456429e+00 	 2.0631933e-01


.. parsed-literal::

      83	 1.3872103e+00	 1.3259822e-01	 1.4395017e+00	 1.3359112e-01	[ 1.2467130e+00]	 2.0228696e-01


.. parsed-literal::

      84	 1.3907613e+00	 1.3213233e-01	 1.4431805e+00	 1.3325628e-01	  1.2441541e+00 	 2.1429944e-01


.. parsed-literal::

      85	 1.3953056e+00	 1.3102907e-01	 1.4479643e+00	 1.3261054e-01	[ 1.2495405e+00]	 2.1650100e-01
      86	 1.4005815e+00	 1.3073622e-01	 1.4532284e+00	 1.3226396e-01	[ 1.2507799e+00]	 1.9770050e-01


.. parsed-literal::

      87	 1.4044531e+00	 1.3043174e-01	 1.4570921e+00	 1.3202108e-01	[ 1.2554823e+00]	 2.1416879e-01


.. parsed-literal::

      88	 1.4103134e+00	 1.2985165e-01	 1.4630789e+00	 1.3177997e-01	[ 1.2648858e+00]	 2.1182847e-01


.. parsed-literal::

      89	 1.4160896e+00	 1.2892208e-01	 1.4691067e+00	 1.3083815e-01	[ 1.2717467e+00]	 2.0439434e-01


.. parsed-literal::

      90	 1.4211273e+00	 1.2837659e-01	 1.4742730e+00	 1.3089708e-01	[ 1.2769110e+00]	 2.1502566e-01


.. parsed-literal::

      91	 1.4240548e+00	 1.2821951e-01	 1.4770554e+00	 1.3064122e-01	[ 1.2781715e+00]	 2.1716547e-01


.. parsed-literal::

      92	 1.4290980e+00	 1.2759479e-01	 1.4822318e+00	 1.2974128e-01	  1.2742365e+00 	 2.1654868e-01


.. parsed-literal::

      93	 1.4331905e+00	 1.2673994e-01	 1.4865076e+00	 1.2924532e-01	  1.2663963e+00 	 2.2225499e-01


.. parsed-literal::

      94	 1.4371187e+00	 1.2639931e-01	 1.4904624e+00	 1.2901206e-01	  1.2719587e+00 	 2.1022129e-01


.. parsed-literal::

      95	 1.4399403e+00	 1.2612880e-01	 1.4932937e+00	 1.2908846e-01	  1.2765785e+00 	 2.0942283e-01


.. parsed-literal::

      96	 1.4434272e+00	 1.2577833e-01	 1.4968981e+00	 1.2962222e-01	  1.2737985e+00 	 2.2285008e-01


.. parsed-literal::

      97	 1.4462603e+00	 1.2549236e-01	 1.4998172e+00	 1.2969996e-01	  1.2775761e+00 	 2.2099400e-01


.. parsed-literal::

      98	 1.4487807e+00	 1.2537398e-01	 1.5023618e+00	 1.2970602e-01	  1.2762102e+00 	 2.0618773e-01
      99	 1.4519561e+00	 1.2517872e-01	 1.5056189e+00	 1.2954246e-01	  1.2735849e+00 	 1.8438792e-01


.. parsed-literal::

     100	 1.4548865e+00	 1.2494465e-01	 1.5085866e+00	 1.2929726e-01	  1.2747561e+00 	 2.0733428e-01
     101	 1.4584668e+00	 1.2457448e-01	 1.5122611e+00	 1.2831270e-01	  1.2711923e+00 	 2.0160508e-01


.. parsed-literal::

     102	 1.4614816e+00	 1.2443467e-01	 1.5152299e+00	 1.2798649e-01	  1.2774539e+00 	 2.1616244e-01


.. parsed-literal::

     103	 1.4635196e+00	 1.2436802e-01	 1.5172306e+00	 1.2779271e-01	[ 1.2797867e+00]	 2.0848894e-01


.. parsed-literal::

     104	 1.4666069e+00	 1.2432540e-01	 1.5204016e+00	 1.2726225e-01	  1.2790799e+00 	 2.1626925e-01


.. parsed-literal::

     105	 1.4685728e+00	 1.2434406e-01	 1.5225068e+00	 1.2708737e-01	  1.2712849e+00 	 2.1619201e-01


.. parsed-literal::

     106	 1.4713531e+00	 1.2437133e-01	 1.5252573e+00	 1.2681380e-01	  1.2745474e+00 	 2.0564294e-01


.. parsed-literal::

     107	 1.4730977e+00	 1.2438787e-01	 1.5270195e+00	 1.2663185e-01	  1.2732870e+00 	 2.2277689e-01


.. parsed-literal::

     108	 1.4746362e+00	 1.2439273e-01	 1.5285618e+00	 1.2657358e-01	  1.2725098e+00 	 2.1929693e-01
     109	 1.4764760e+00	 1.2437988e-01	 1.5305617e+00	 1.2622632e-01	  1.2647947e+00 	 1.9202662e-01


.. parsed-literal::

     110	 1.4792381e+00	 1.2430245e-01	 1.5332456e+00	 1.2625804e-01	  1.2674921e+00 	 2.1147227e-01


.. parsed-literal::

     111	 1.4802447e+00	 1.2423191e-01	 1.5342549e+00	 1.2620604e-01	  1.2686922e+00 	 2.1366429e-01
     112	 1.4818611e+00	 1.2411921e-01	 1.5359378e+00	 1.2610861e-01	  1.2670845e+00 	 1.9485164e-01


.. parsed-literal::

     113	 1.4839023e+00	 1.2401841e-01	 1.5380709e+00	 1.2593072e-01	  1.2647978e+00 	 2.0335937e-01


.. parsed-literal::

     114	 1.4853986e+00	 1.2397254e-01	 1.5398499e+00	 1.2575562e-01	  1.2434127e+00 	 2.1565866e-01
     115	 1.4880916e+00	 1.2391208e-01	 1.5424132e+00	 1.2560340e-01	  1.2501308e+00 	 1.9923759e-01


.. parsed-literal::

     116	 1.4893318e+00	 1.2389576e-01	 1.5436177e+00	 1.2550953e-01	  1.2486614e+00 	 2.1985793e-01


.. parsed-literal::

     117	 1.4915631e+00	 1.2384470e-01	 1.5458377e+00	 1.2533626e-01	  1.2411151e+00 	 2.1852493e-01


.. parsed-literal::

     118	 1.4928472e+00	 1.2375463e-01	 1.5471754e+00	 1.2509046e-01	  1.2299998e+00 	 2.9419160e-01
     119	 1.4945453e+00	 1.2366986e-01	 1.5488753e+00	 1.2498181e-01	  1.2274039e+00 	 1.9360924e-01


.. parsed-literal::

     120	 1.4964631e+00	 1.2352016e-01	 1.5508630e+00	 1.2484345e-01	  1.2212650e+00 	 1.9084072e-01


.. parsed-literal::

     121	 1.4984975e+00	 1.2337601e-01	 1.5530307e+00	 1.2469454e-01	  1.2175176e+00 	 2.1886683e-01


.. parsed-literal::

     122	 1.5004469e+00	 1.2319995e-01	 1.5550143e+00	 1.2456551e-01	  1.2098115e+00 	 2.2106862e-01
     123	 1.5020664e+00	 1.2312682e-01	 1.5566528e+00	 1.2444109e-01	  1.2020762e+00 	 1.7856646e-01


.. parsed-literal::

     124	 1.5034470e+00	 1.2304666e-01	 1.5580375e+00	 1.2438049e-01	  1.1961908e+00 	 2.1347284e-01


.. parsed-literal::

     125	 1.5045207e+00	 1.2303666e-01	 1.5591518e+00	 1.2409884e-01	  1.1902670e+00 	 2.1293783e-01
     126	 1.5058646e+00	 1.2297713e-01	 1.5604596e+00	 1.2404385e-01	  1.1919595e+00 	 1.9578218e-01


.. parsed-literal::

     127	 1.5073736e+00	 1.2290330e-01	 1.5619658e+00	 1.2394004e-01	  1.1936340e+00 	 1.8829107e-01


.. parsed-literal::

     128	 1.5086389e+00	 1.2284475e-01	 1.5632666e+00	 1.2380841e-01	  1.1914480e+00 	 2.1044993e-01


.. parsed-literal::

     129	 1.5111537e+00	 1.2276745e-01	 1.5659763e+00	 1.2364010e-01	  1.1730063e+00 	 2.0576596e-01


.. parsed-literal::

     130	 1.5125709e+00	 1.2273926e-01	 1.5674326e+00	 1.2356496e-01	  1.1688795e+00 	 3.2290030e-01


.. parsed-literal::

     131	 1.5137855e+00	 1.2271519e-01	 1.5686700e+00	 1.2351211e-01	  1.1599816e+00 	 2.1308517e-01


.. parsed-literal::

     132	 1.5150060e+00	 1.2266522e-01	 1.5699410e+00	 1.2354601e-01	  1.1484605e+00 	 2.0784831e-01


.. parsed-literal::

     133	 1.5166248e+00	 1.2256499e-01	 1.5716128e+00	 1.2353977e-01	  1.1332778e+00 	 2.0266914e-01
     134	 1.5186674e+00	 1.2247284e-01	 1.5737172e+00	 1.2378167e-01	  1.1160426e+00 	 1.7814755e-01


.. parsed-literal::

     135	 1.5202011e+00	 1.2234033e-01	 1.5752237e+00	 1.2385096e-01	  1.1028689e+00 	 2.1442175e-01


.. parsed-literal::

     136	 1.5216134e+00	 1.2231514e-01	 1.5765810e+00	 1.2396112e-01	  1.1017958e+00 	 2.1564031e-01


.. parsed-literal::

     137	 1.5230352e+00	 1.2222626e-01	 1.5780029e+00	 1.2402222e-01	  1.0956966e+00 	 2.1174312e-01


.. parsed-literal::

     138	 1.5240556e+00	 1.2215074e-01	 1.5790818e+00	 1.2412158e-01	  1.0936557e+00 	 2.0469761e-01


.. parsed-literal::

     139	 1.5251523e+00	 1.2207428e-01	 1.5801941e+00	 1.2410688e-01	  1.0889962e+00 	 2.1755290e-01


.. parsed-literal::

     140	 1.5262682e+00	 1.2195444e-01	 1.5813759e+00	 1.2414625e-01	  1.0805959e+00 	 2.2101092e-01


.. parsed-literal::

     141	 1.5272376e+00	 1.2184047e-01	 1.5823844e+00	 1.2424706e-01	  1.0755983e+00 	 2.2134304e-01


.. parsed-literal::

     142	 1.5281776e+00	 1.2164717e-01	 1.5834552e+00	 1.2432331e-01	  1.0748270e+00 	 2.1563840e-01


.. parsed-literal::

     143	 1.5293960e+00	 1.2159940e-01	 1.5846173e+00	 1.2449033e-01	  1.0703197e+00 	 2.2083354e-01


.. parsed-literal::

     144	 1.5299783e+00	 1.2160335e-01	 1.5851620e+00	 1.2450354e-01	  1.0691905e+00 	 2.1479630e-01


.. parsed-literal::

     145	 1.5312948e+00	 1.2153352e-01	 1.5864748e+00	 1.2450554e-01	  1.0610768e+00 	 2.1175599e-01


.. parsed-literal::

     146	 1.5316599e+00	 1.2150557e-01	 1.5870040e+00	 1.2459410e-01	  1.0472795e+00 	 2.1165681e-01
     147	 1.5329977e+00	 1.2143801e-01	 1.5882340e+00	 1.2448162e-01	  1.0452904e+00 	 1.9894028e-01


.. parsed-literal::

     148	 1.5335848e+00	 1.2138593e-01	 1.5888360e+00	 1.2443311e-01	  1.0400621e+00 	 2.1738768e-01


.. parsed-literal::

     149	 1.5345774e+00	 1.2130748e-01	 1.5898749e+00	 1.2434823e-01	  1.0264255e+00 	 2.0611024e-01


.. parsed-literal::

     150	 1.5356454e+00	 1.2125710e-01	 1.5910546e+00	 1.2412328e-01	  1.0040763e+00 	 2.2010303e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 7s, sys: 1.2 s, total: 2min 8s
    Wall time: 32.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fd224a24be0>



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
    CPU times: user 1.85 s, sys: 53 ms, total: 1.9 s
    Wall time: 614 ms


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

