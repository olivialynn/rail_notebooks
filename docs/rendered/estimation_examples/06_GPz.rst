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
       1	-3.4102477e-01	 3.1960677e-01	-3.3134460e-01	 3.2420183e-01	[-3.4047273e-01]	 4.6154857e-01


.. parsed-literal::

       2	-2.6928791e-01	 3.0886319e-01	-2.4528154e-01	 3.1353757e-01	[-2.6005501e-01]	 2.3239517e-01


.. parsed-literal::

       3	-2.2371811e-01	 2.8762058e-01	-1.8103369e-01	 2.9198717e-01	[-2.0162403e-01]	 2.8191042e-01


.. parsed-literal::

       4	-1.9392968e-01	 2.6382410e-01	-1.5317071e-01	 2.7023732e-01	[-1.9151562e-01]	 2.1249533e-01
       5	-1.0143634e-01	 2.5654083e-01	-6.6856289e-02	 2.6057208e-01	[-8.7783286e-02]	 1.8704486e-01


.. parsed-literal::

       6	-6.8618152e-02	 2.5082827e-01	-3.7141503e-02	 2.5510856e-01	[-5.3031121e-02]	 2.2073603e-01
       7	-5.1312590e-02	 2.4843666e-01	-2.6782220e-02	 2.5250917e-01	[-4.2763541e-02]	 1.7319608e-01


.. parsed-literal::

       8	-3.8305722e-02	 2.4628851e-01	-1.7788473e-02	 2.4973831e-01	[-3.1724592e-02]	 2.0782161e-01


.. parsed-literal::

       9	-2.3966368e-02	 2.4363006e-01	-6.4419260e-03	 2.4587029e-01	[-1.5710251e-02]	 2.0729327e-01


.. parsed-literal::

      10	-1.3836171e-02	 2.4181150e-01	 1.6988371e-03	 2.4484835e-01	[-1.1302191e-02]	 2.1320391e-01


.. parsed-literal::

      11	-1.0091509e-02	 2.4127670e-01	 4.3054014e-03	 2.4394627e-01	[-7.6740476e-03]	 2.1041107e-01


.. parsed-literal::

      12	-7.0394931e-03	 2.4065447e-01	 7.1400023e-03	 2.4328242e-01	[-4.0224007e-03]	 2.0987821e-01
      13	-4.3488987e-03	 2.4014584e-01	 9.6624771e-03	 2.4276028e-01	[-1.4680316e-03]	 1.7966366e-01


.. parsed-literal::

      14	 2.0581165e-04	 2.3922473e-01	 1.4512335e-02	 2.4184800e-01	[ 3.3784702e-03]	 1.9839668e-01


.. parsed-literal::

      15	 1.1535029e-01	 2.2520336e-01	 1.3613889e-01	 2.2958037e-01	[ 1.2496240e-01]	 2.9276252e-01
      16	 2.1002766e-01	 2.2052588e-01	 2.3636996e-01	 2.2685506e-01	[ 2.2173044e-01]	 1.9908476e-01


.. parsed-literal::

      17	 2.7078636e-01	 2.1969867e-01	 3.0207324e-01	 2.2873036e-01	[ 2.7130200e-01]	 2.0390296e-01
      18	 3.2105266e-01	 2.1356544e-01	 3.5488467e-01	 2.2708864e-01	[ 3.0714936e-01]	 1.7486620e-01


.. parsed-literal::

      19	 3.7240438e-01	 2.0873118e-01	 4.0624752e-01	 2.2104706e-01	[ 3.6474301e-01]	 1.9159126e-01
      20	 4.2326349e-01	 2.0638243e-01	 4.5674662e-01	 2.1729930e-01	[ 4.2592131e-01]	 1.7423940e-01


.. parsed-literal::

      21	 4.8804383e-01	 2.0470913e-01	 5.2251281e-01	 2.1467276e-01	[ 4.9693075e-01]	 1.9144464e-01


.. parsed-literal::

      22	 5.7201204e-01	 2.0226615e-01	 6.0986304e-01	 2.1110995e-01	[ 5.8540335e-01]	 2.0816469e-01


.. parsed-literal::

      23	 6.1788001e-01	 2.0381235e-01	 6.5770217e-01	 2.1432721e-01	[ 6.2279962e-01]	 2.1368074e-01


.. parsed-literal::

      24	 6.5444028e-01	 1.9765980e-01	 6.9293140e-01	 2.0879620e-01	[ 6.5931118e-01]	 2.0775962e-01
      25	 6.8044552e-01	 1.9716049e-01	 7.1899756e-01	 2.0934518e-01	[ 6.8241708e-01]	 1.7599869e-01


.. parsed-literal::

      26	 7.1789336e-01	 1.9879956e-01	 7.5555094e-01	 2.1096628e-01	[ 7.1713601e-01]	 3.0804610e-01


.. parsed-literal::

      27	 7.4647107e-01	 1.9984986e-01	 7.8410821e-01	 2.1162746e-01	[ 7.4068490e-01]	 3.1427884e-01


.. parsed-literal::

      28	 7.7729182e-01	 1.9980784e-01	 8.1584039e-01	 2.1170155e-01	[ 7.5935936e-01]	 2.1361876e-01


.. parsed-literal::

      29	 8.0088074e-01	 1.9726146e-01	 8.4126612e-01	 2.0994179e-01	[ 7.8359571e-01]	 2.0455909e-01
      30	 8.2749243e-01	 1.9579576e-01	 8.6696484e-01	 2.0750739e-01	[ 8.1414581e-01]	 1.9212461e-01


.. parsed-literal::

      31	 8.5114831e-01	 1.9539292e-01	 8.9154627e-01	 2.0688614e-01	[ 8.3406943e-01]	 2.0792890e-01
      32	 8.8776035e-01	 1.9451676e-01	 9.2920912e-01	 2.0638604e-01	[ 8.6775462e-01]	 1.8965435e-01


.. parsed-literal::

      33	 9.2032385e-01	 1.8983107e-01	 9.6424796e-01	 2.0313384e-01	[ 9.0147325e-01]	 2.0575428e-01


.. parsed-literal::

      34	 9.4520431e-01	 1.8784479e-01	 9.8939072e-01	 2.0101304e-01	[ 9.3388852e-01]	 2.0991325e-01


.. parsed-literal::

      35	 9.6269299e-01	 1.8661941e-01	 1.0068773e+00	 1.9936426e-01	[ 9.4667532e-01]	 2.0406199e-01
      36	 9.8670186e-01	 1.8541645e-01	 1.0314908e+00	 1.9695305e-01	[ 9.6359648e-01]	 1.7812443e-01


.. parsed-literal::

      37	 1.0035924e+00	 1.8361942e-01	 1.0492349e+00	 1.9421254e-01	[ 9.7978876e-01]	 1.9494319e-01
      38	 1.0192638e+00	 1.8100665e-01	 1.0655789e+00	 1.9098233e-01	[ 9.9320213e-01]	 1.9894290e-01


.. parsed-literal::

      39	 1.0279699e+00	 1.8012837e-01	 1.0742769e+00	 1.9035964e-01	[ 1.0018979e+00]	 2.1275592e-01
      40	 1.0435849e+00	 1.7835628e-01	 1.0907544e+00	 1.8865800e-01	[ 1.0161312e+00]	 1.9994402e-01


.. parsed-literal::

      41	 1.0577542e+00	 1.7737280e-01	 1.1051866e+00	 1.8772688e-01	[ 1.0297912e+00]	 2.1048307e-01


.. parsed-literal::

      42	 1.0769551e+00	 1.7545163e-01	 1.1250899e+00	 1.8566349e-01	[ 1.0475064e+00]	 2.0322537e-01


.. parsed-literal::

      43	 1.0911627e+00	 1.7379951e-01	 1.1404554e+00	 1.8415395e-01	[ 1.0563193e+00]	 2.1120596e-01


.. parsed-literal::

      44	 1.1061719e+00	 1.7373229e-01	 1.1547964e+00	 1.8396549e-01	[ 1.0733521e+00]	 2.0363617e-01
      45	 1.1165159e+00	 1.7224010e-01	 1.1649797e+00	 1.8262195e-01	[ 1.0830793e+00]	 1.8493176e-01


.. parsed-literal::

      46	 1.1347024e+00	 1.6986025e-01	 1.1834243e+00	 1.8100986e-01	[ 1.0999076e+00]	 2.2132444e-01


.. parsed-literal::

      47	 1.1452375e+00	 1.6206505e-01	 1.1934893e+00	 1.7528602e-01	[ 1.1071512e+00]	 2.1296787e-01
      48	 1.1611293e+00	 1.6284530e-01	 1.2098265e+00	 1.7561322e-01	[ 1.1226593e+00]	 1.9737554e-01


.. parsed-literal::

      49	 1.1725550e+00	 1.6276283e-01	 1.2220795e+00	 1.7529524e-01	[ 1.1280410e+00]	 2.1014404e-01


.. parsed-literal::

      50	 1.1849151e+00	 1.6124841e-01	 1.2349758e+00	 1.7332557e-01	[ 1.1353483e+00]	 2.1629977e-01


.. parsed-literal::

      51	 1.2011979e+00	 1.5806140e-01	 1.2523403e+00	 1.6986644e-01	[ 1.1437417e+00]	 2.0771217e-01


.. parsed-literal::

      52	 1.2128864e+00	 1.5661810e-01	 1.2632761e+00	 1.6799031e-01	[ 1.1587038e+00]	 2.1674013e-01


.. parsed-literal::

      53	 1.2192162e+00	 1.5559756e-01	 1.2694084e+00	 1.6716274e-01	[ 1.1646385e+00]	 2.1433234e-01
      54	 1.2325366e+00	 1.5230089e-01	 1.2829790e+00	 1.6407552e-01	[ 1.1730002e+00]	 1.8550730e-01


.. parsed-literal::

      55	 1.2423832e+00	 1.5068873e-01	 1.2932553e+00	 1.6238461e-01	[ 1.1830606e+00]	 2.0978260e-01
      56	 1.2527770e+00	 1.4912821e-01	 1.3035945e+00	 1.6097327e-01	[ 1.1934185e+00]	 1.9684672e-01


.. parsed-literal::

      57	 1.2621291e+00	 1.4859903e-01	 1.3131753e+00	 1.6068559e-01	[ 1.2006605e+00]	 2.0879221e-01


.. parsed-literal::

      58	 1.2702042e+00	 1.4815627e-01	 1.3214460e+00	 1.6033123e-01	[ 1.2057839e+00]	 2.1289325e-01


.. parsed-literal::

      59	 1.2817161e+00	 1.4911601e-01	 1.3333916e+00	 1.6203431e-01	[ 1.2150714e+00]	 2.1200681e-01


.. parsed-literal::

      60	 1.2920874e+00	 1.4793212e-01	 1.3436849e+00	 1.6159823e-01	[ 1.2165615e+00]	 2.0453191e-01


.. parsed-literal::

      61	 1.2991750e+00	 1.4747926e-01	 1.3505118e+00	 1.6131526e-01	[ 1.2259844e+00]	 2.0171666e-01
      62	 1.3083085e+00	 1.4666238e-01	 1.3596918e+00	 1.6098387e-01	[ 1.2352260e+00]	 1.9275165e-01


.. parsed-literal::

      63	 1.3164047e+00	 1.4581304e-01	 1.3676286e+00	 1.6049898e-01	[ 1.2432769e+00]	 2.1450663e-01
      64	 1.3279769e+00	 1.4278533e-01	 1.3798278e+00	 1.5767151e-01	  1.2414475e+00 	 1.8577290e-01


.. parsed-literal::

      65	 1.3386426e+00	 1.4098634e-01	 1.3904911e+00	 1.5611778e-01	[ 1.2447849e+00]	 1.8848205e-01


.. parsed-literal::

      66	 1.3450695e+00	 1.4092204e-01	 1.3967531e+00	 1.5589467e-01	[ 1.2498748e+00]	 2.1452308e-01


.. parsed-literal::

      67	 1.3528250e+00	 1.4005252e-01	 1.4048168e+00	 1.5520576e-01	  1.2486851e+00 	 2.0787835e-01
      68	 1.3605599e+00	 1.3860636e-01	 1.4131416e+00	 1.5348696e-01	  1.2448919e+00 	 1.9610667e-01


.. parsed-literal::

      69	 1.3658982e+00	 1.3753450e-01	 1.4186922e+00	 1.5282081e-01	  1.2445930e+00 	 2.0299888e-01


.. parsed-literal::

      70	 1.3712912e+00	 1.3738996e-01	 1.4239179e+00	 1.5276540e-01	[ 1.2529700e+00]	 2.2616100e-01


.. parsed-literal::

      71	 1.3775502e+00	 1.3703677e-01	 1.4302938e+00	 1.5247324e-01	[ 1.2601119e+00]	 2.0970249e-01
      72	 1.3841783e+00	 1.3665719e-01	 1.4371257e+00	 1.5223459e-01	[ 1.2638651e+00]	 1.8784070e-01


.. parsed-literal::

      73	 1.3920340e+00	 1.3500152e-01	 1.4452251e+00	 1.5123883e-01	  1.2609284e+00 	 2.0367336e-01
      74	 1.3995503e+00	 1.3485227e-01	 1.4527775e+00	 1.5150201e-01	[ 1.2644279e+00]	 1.8656278e-01


.. parsed-literal::

      75	 1.4037999e+00	 1.3446305e-01	 1.4568725e+00	 1.5141389e-01	[ 1.2662896e+00]	 2.1434450e-01
      76	 1.4098991e+00	 1.3280878e-01	 1.4631157e+00	 1.5056333e-01	  1.2641351e+00 	 2.0123601e-01


.. parsed-literal::

      77	 1.4137992e+00	 1.3155076e-01	 1.4672200e+00	 1.5023732e-01	  1.2612893e+00 	 1.8535781e-01


.. parsed-literal::

      78	 1.4188468e+00	 1.3101769e-01	 1.4722177e+00	 1.4976970e-01	[ 1.2665477e+00]	 2.1206450e-01


.. parsed-literal::

      79	 1.4239624e+00	 1.3033752e-01	 1.4773285e+00	 1.4911650e-01	[ 1.2747392e+00]	 2.0196915e-01


.. parsed-literal::

      80	 1.4275864e+00	 1.2985329e-01	 1.4809210e+00	 1.4870986e-01	[ 1.2800287e+00]	 2.2199607e-01
      81	 1.4331214e+00	 1.2852524e-01	 1.4867851e+00	 1.4775401e-01	  1.2774034e+00 	 1.8046045e-01


.. parsed-literal::

      82	 1.4370666e+00	 1.2845297e-01	 1.4906636e+00	 1.4765947e-01	[ 1.2819447e+00]	 2.0314407e-01


.. parsed-literal::

      83	 1.4396474e+00	 1.2824011e-01	 1.4931534e+00	 1.4771337e-01	[ 1.2819683e+00]	 2.0690036e-01
      84	 1.4421286e+00	 1.2778428e-01	 1.4956596e+00	 1.4756191e-01	  1.2803780e+00 	 1.8052173e-01


.. parsed-literal::

      85	 1.4455243e+00	 1.2729640e-01	 1.4991644e+00	 1.4739862e-01	  1.2779142e+00 	 2.2084641e-01
      86	 1.4498424e+00	 1.2666339e-01	 1.5036486e+00	 1.4698569e-01	  1.2779110e+00 	 1.8144870e-01


.. parsed-literal::

      87	 1.4534487e+00	 1.2602466e-01	 1.5074593e+00	 1.4647360e-01	  1.2797991e+00 	 2.0552921e-01
      88	 1.4566906e+00	 1.2585933e-01	 1.5106927e+00	 1.4617261e-01	[ 1.2845283e+00]	 1.8244338e-01


.. parsed-literal::

      89	 1.4595135e+00	 1.2560810e-01	 1.5135129e+00	 1.4588603e-01	[ 1.2866239e+00]	 1.9989181e-01


.. parsed-literal::

      90	 1.4629433e+00	 1.2527488e-01	 1.5169376e+00	 1.4559655e-01	[ 1.2874796e+00]	 2.2118425e-01


.. parsed-literal::

      91	 1.4665322e+00	 1.2480924e-01	 1.5205018e+00	 1.4568190e-01	  1.2830062e+00 	 2.0846009e-01


.. parsed-literal::

      92	 1.4689424e+00	 1.2460434e-01	 1.5228364e+00	 1.4565463e-01	  1.2812683e+00 	 2.0874691e-01


.. parsed-literal::

      93	 1.4721931e+00	 1.2437131e-01	 1.5260543e+00	 1.4566387e-01	  1.2757977e+00 	 2.1287990e-01


.. parsed-literal::

      94	 1.4747637e+00	 1.2399006e-01	 1.5287328e+00	 1.4534391e-01	  1.2675386e+00 	 2.0784307e-01


.. parsed-literal::

      95	 1.4779057e+00	 1.2393896e-01	 1.5318716e+00	 1.4542300e-01	  1.2667921e+00 	 2.1188116e-01
      96	 1.4806548e+00	 1.2381959e-01	 1.5346832e+00	 1.4541881e-01	  1.2667131e+00 	 1.9125080e-01


.. parsed-literal::

      97	 1.4827914e+00	 1.2360977e-01	 1.5368869e+00	 1.4547535e-01	  1.2664385e+00 	 2.0671487e-01


.. parsed-literal::

      98	 1.4854531e+00	 1.2327536e-01	 1.5395767e+00	 1.4559652e-01	  1.2658423e+00 	 2.0859861e-01
      99	 1.4884324e+00	 1.2274968e-01	 1.5426571e+00	 1.4569762e-01	  1.2630809e+00 	 1.9951582e-01


.. parsed-literal::

     100	 1.4905503e+00	 1.2241919e-01	 1.5448274e+00	 1.4594970e-01	  1.2557356e+00 	 2.0183897e-01


.. parsed-literal::

     101	 1.4924295e+00	 1.2232960e-01	 1.5466768e+00	 1.4585169e-01	  1.2557375e+00 	 2.0551014e-01


.. parsed-literal::

     102	 1.4948536e+00	 1.2217359e-01	 1.5491375e+00	 1.4563152e-01	  1.2546114e+00 	 2.1031833e-01
     103	 1.4972651e+00	 1.2197436e-01	 1.5516168e+00	 1.4549686e-01	  1.2549430e+00 	 1.8329144e-01


.. parsed-literal::

     104	 1.4994239e+00	 1.2156774e-01	 1.5539180e+00	 1.4559457e-01	  1.2549664e+00 	 2.1573639e-01


.. parsed-literal::

     105	 1.5019578e+00	 1.2149690e-01	 1.5563758e+00	 1.4556600e-01	  1.2626983e+00 	 2.2320747e-01


.. parsed-literal::

     106	 1.5033123e+00	 1.2136675e-01	 1.5577051e+00	 1.4577040e-01	  1.2663374e+00 	 2.1207666e-01


.. parsed-literal::

     107	 1.5058227e+00	 1.2113531e-01	 1.5602433e+00	 1.4605587e-01	  1.2694594e+00 	 2.1833348e-01


.. parsed-literal::

     108	 1.5085354e+00	 1.2095335e-01	 1.5630716e+00	 1.4676839e-01	  1.2701571e+00 	 2.0507789e-01
     109	 1.5110672e+00	 1.2073422e-01	 1.5656374e+00	 1.4651277e-01	  1.2708387e+00 	 1.8775535e-01


.. parsed-literal::

     110	 1.5124890e+00	 1.2071750e-01	 1.5670425e+00	 1.4625336e-01	  1.2692019e+00 	 1.9050813e-01


.. parsed-literal::

     111	 1.5143403e+00	 1.2068690e-01	 1.5689369e+00	 1.4592967e-01	  1.2659135e+00 	 2.0713496e-01


.. parsed-literal::

     112	 1.5165057e+00	 1.2057463e-01	 1.5711523e+00	 1.4552896e-01	  1.2661091e+00 	 2.2108626e-01
     113	 1.5188274e+00	 1.2038547e-01	 1.5735397e+00	 1.4530747e-01	  1.2670153e+00 	 1.9403863e-01


.. parsed-literal::

     114	 1.5206998e+00	 1.2025251e-01	 1.5753632e+00	 1.4522045e-01	  1.2728191e+00 	 2.0361638e-01


.. parsed-literal::

     115	 1.5225068e+00	 1.2010301e-01	 1.5771047e+00	 1.4536014e-01	  1.2789441e+00 	 2.1846795e-01


.. parsed-literal::

     116	 1.5241273e+00	 1.1992165e-01	 1.5787358e+00	 1.4551170e-01	  1.2815151e+00 	 2.0883131e-01
     117	 1.5265709e+00	 1.1970142e-01	 1.5812472e+00	 1.4564093e-01	  1.2815141e+00 	 2.0462728e-01


.. parsed-literal::

     118	 1.5281400e+00	 1.1934383e-01	 1.5830674e+00	 1.4592093e-01	  1.2756091e+00 	 2.0302820e-01


.. parsed-literal::

     119	 1.5310205e+00	 1.1936205e-01	 1.5859208e+00	 1.4563540e-01	  1.2767643e+00 	 2.1440315e-01


.. parsed-literal::

     120	 1.5324054e+00	 1.1937036e-01	 1.5873151e+00	 1.4543474e-01	  1.2775460e+00 	 2.0911193e-01


.. parsed-literal::

     121	 1.5340984e+00	 1.1931087e-01	 1.5890735e+00	 1.4524847e-01	  1.2781106e+00 	 2.1481848e-01


.. parsed-literal::

     122	 1.5355997e+00	 1.1929224e-01	 1.5906191e+00	 1.4524130e-01	  1.2834194e+00 	 2.0503306e-01


.. parsed-literal::

     123	 1.5374205e+00	 1.1919123e-01	 1.5924102e+00	 1.4520211e-01	  1.2846650e+00 	 2.1025944e-01
     124	 1.5386828e+00	 1.1905159e-01	 1.5936586e+00	 1.4520479e-01	  1.2859835e+00 	 1.7779112e-01


.. parsed-literal::

     125	 1.5400273e+00	 1.1893634e-01	 1.5950095e+00	 1.4520396e-01	  1.2859005e+00 	 2.0597053e-01
     126	 1.5414966e+00	 1.1865200e-01	 1.5965849e+00	 1.4484778e-01	  1.2842359e+00 	 2.1008849e-01


.. parsed-literal::

     127	 1.5436100e+00	 1.1865201e-01	 1.5986588e+00	 1.4482798e-01	  1.2834506e+00 	 2.0306110e-01


.. parsed-literal::

     128	 1.5452627e+00	 1.1862270e-01	 1.6003330e+00	 1.4464938e-01	  1.2818480e+00 	 2.0298743e-01


.. parsed-literal::

     129	 1.5470274e+00	 1.1854965e-01	 1.6021064e+00	 1.4436197e-01	  1.2813948e+00 	 2.2006130e-01


.. parsed-literal::

     130	 1.5493055e+00	 1.1845416e-01	 1.6044021e+00	 1.4404284e-01	  1.2796292e+00 	 2.1411705e-01
     131	 1.5512814e+00	 1.1837526e-01	 1.6063473e+00	 1.4353232e-01	  1.2799727e+00 	 1.8051481e-01


.. parsed-literal::

     132	 1.5528418e+00	 1.1832448e-01	 1.6078622e+00	 1.4363876e-01	  1.2814045e+00 	 2.2162366e-01
     133	 1.5545545e+00	 1.1823786e-01	 1.6096109e+00	 1.4373434e-01	  1.2796606e+00 	 1.8035483e-01


.. parsed-literal::

     134	 1.5562421e+00	 1.1813755e-01	 1.6114142e+00	 1.4355377e-01	  1.2735032e+00 	 2.1559167e-01


.. parsed-literal::

     135	 1.5582532e+00	 1.1796226e-01	 1.6136490e+00	 1.4334046e-01	  1.2593332e+00 	 2.0897341e-01


.. parsed-literal::

     136	 1.5596068e+00	 1.1782334e-01	 1.6150929e+00	 1.4315377e-01	  1.2519028e+00 	 2.1281481e-01


.. parsed-literal::

     137	 1.5609835e+00	 1.1771124e-01	 1.6164348e+00	 1.4301444e-01	  1.2515847e+00 	 2.0921469e-01


.. parsed-literal::

     138	 1.5626351e+00	 1.1756302e-01	 1.6180511e+00	 1.4299444e-01	  1.2525006e+00 	 2.1837306e-01


.. parsed-literal::

     139	 1.5639626e+00	 1.1738065e-01	 1.6194240e+00	 1.4268580e-01	  1.2507671e+00 	 2.1093345e-01


.. parsed-literal::

     140	 1.5653815e+00	 1.1734371e-01	 1.6207920e+00	 1.4275228e-01	  1.2514255e+00 	 2.0763898e-01


.. parsed-literal::

     141	 1.5663714e+00	 1.1730159e-01	 1.6217854e+00	 1.4266911e-01	  1.2496845e+00 	 2.0821095e-01
     142	 1.5675108e+00	 1.1724311e-01	 1.6229453e+00	 1.4250913e-01	  1.2457396e+00 	 1.9522738e-01


.. parsed-literal::

     143	 1.5685798e+00	 1.1711624e-01	 1.6241365e+00	 1.4208479e-01	  1.2346867e+00 	 1.8327451e-01
     144	 1.5701332e+00	 1.1710388e-01	 1.6256813e+00	 1.4206928e-01	  1.2292748e+00 	 2.0126820e-01


.. parsed-literal::

     145	 1.5708637e+00	 1.1706688e-01	 1.6264227e+00	 1.4203299e-01	  1.2260767e+00 	 2.0901537e-01


.. parsed-literal::

     146	 1.5718907e+00	 1.1699970e-01	 1.6274733e+00	 1.4194306e-01	  1.2211879e+00 	 2.2086859e-01
     147	 1.5730190e+00	 1.1688855e-01	 1.6286170e+00	 1.4180607e-01	  1.2136358e+00 	 1.8688083e-01


.. parsed-literal::

     148	 1.5744357e+00	 1.1682757e-01	 1.6300095e+00	 1.4164579e-01	  1.2080440e+00 	 2.1185279e-01
     149	 1.5757602e+00	 1.1679446e-01	 1.6312730e+00	 1.4148791e-01	  1.2046264e+00 	 1.7334890e-01


.. parsed-literal::

     150	 1.5770754e+00	 1.1678034e-01	 1.6325444e+00	 1.4123104e-01	  1.2001123e+00 	 2.1180153e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.1 s, total: 2min 4s
    Wall time: 31.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f87ba388c10>



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
    CPU times: user 1.79 s, sys: 42 ms, total: 1.83 s
    Wall time: 578 ms


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

