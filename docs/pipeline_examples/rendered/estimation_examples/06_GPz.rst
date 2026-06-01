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
       1	-3.4256259e-01	 3.2024033e-01	-3.3226429e-01	 3.2224079e-01	[-3.3551109e-01]	 4.5894194e-01


.. parsed-literal::

       2	-2.7501828e-01	 3.1098707e-01	-2.5211161e-01	 3.1332679e-01	[-2.5881484e-01]	 2.2753167e-01


.. parsed-literal::

       3	-2.3325802e-01	 2.9149596e-01	-1.9338456e-01	 2.9297744e-01	[-2.0129497e-01]	 2.7407479e-01
       4	-1.8568702e-01	 2.6759451e-01	-1.4137947e-01	 2.6655482e-01	[-1.3730286e-01]	 1.9289112e-01


.. parsed-literal::

       5	-1.1155976e-01	 2.5735376e-01	-8.1663871e-02	 2.5794173e-01	[-8.5734487e-02]	 2.0452476e-01


.. parsed-literal::

       6	-7.1982629e-02	 2.5200325e-01	-4.3624966e-02	 2.5214102e-01	[-4.4699642e-02]	 2.0529437e-01


.. parsed-literal::

       7	-5.2536041e-02	 2.4882620e-01	-2.8180741e-02	 2.4921600e-01	[-3.0345852e-02]	 2.0948195e-01


.. parsed-literal::

       8	-3.3490681e-02	 2.4521469e-01	-1.3120370e-02	 2.4715955e-01	[-2.1926285e-02]	 2.0939445e-01


.. parsed-literal::

       9	-2.2518459e-02	 2.4302066e-01	-4.0416039e-03	 2.4573302e-01	[-1.5745622e-02]	 2.0444298e-01
      10	-1.3827341e-02	 2.4155709e-01	 2.5821770e-03	 2.4399098e-01	[-7.3020758e-03]	 1.9502711e-01


.. parsed-literal::

      11	-7.1526247e-03	 2.4043474e-01	 8.1610371e-03	 2.4364351e-01	[-5.0101633e-03]	 2.0205951e-01
      12	-4.7297107e-03	 2.4001459e-01	 1.0304042e-02	 2.4379632e-01	[-4.4583067e-03]	 2.0056629e-01


.. parsed-literal::

      13	-8.5253150e-04	 2.3934887e-01	 1.3826856e-02	 2.4295327e-01	[-9.6960236e-04]	 2.1182108e-01
      14	 5.7526359e-03	 2.3799544e-01	 2.1220508e-02	 2.4239065e-01	[ 4.4389898e-03]	 1.8390560e-01


.. parsed-literal::

      15	 8.1929266e-02	 2.2719334e-01	 1.0090034e-01	 2.2813045e-01	[ 9.3862841e-02]	 4.3854165e-01
      16	 1.2858116e-01	 2.2419590e-01	 1.5062026e-01	 2.2718208e-01	[ 1.3808658e-01]	 2.0877600e-01


.. parsed-literal::

      17	 2.5022892e-01	 2.2432918e-01	 2.7917088e-01	 2.2601804e-01	[ 2.6490198e-01]	 2.0438480e-01


.. parsed-literal::

      18	 3.2655170e-01	 2.1985878e-01	 3.6003067e-01	 2.2613596e-01	[ 3.2368323e-01]	 2.0511794e-01
      19	 3.6980251e-01	 2.1897647e-01	 4.0299202e-01	 2.2405004e-01	[ 3.7835595e-01]	 2.0316601e-01


.. parsed-literal::

      20	 4.1289119e-01	 2.1550190e-01	 4.4664203e-01	 2.1993483e-01	[ 4.1935835e-01]	 2.1254563e-01


.. parsed-literal::

      21	 4.6358918e-01	 2.1367463e-01	 4.9890648e-01	 2.1875826e-01	[ 4.7277439e-01]	 2.0705819e-01


.. parsed-literal::

      22	 5.3068732e-01	 2.1226158e-01	 5.7063246e-01	 2.1859489e-01	[ 5.5245091e-01]	 2.0437670e-01
      23	 5.6159576e-01	 2.1265779e-01	 6.0311319e-01	 2.1820880e-01	[ 5.9726273e-01]	 1.9147587e-01


.. parsed-literal::

      24	 5.9189688e-01	 2.0901338e-01	 6.3174805e-01	 2.1537450e-01	[ 6.2036550e-01]	 2.1264887e-01


.. parsed-literal::

      25	 6.2811128e-01	 2.1128078e-01	 6.6751204e-01	 2.1846756e-01	[ 6.6082948e-01]	 2.1012759e-01


.. parsed-literal::

      26	 6.5929269e-01	 2.1145999e-01	 6.9680159e-01	 2.1666870e-01	[ 6.8863454e-01]	 2.1812201e-01


.. parsed-literal::

      27	 6.9652542e-01	 2.1031095e-01	 7.3603438e-01	 2.1564936e-01	[ 7.3411399e-01]	 2.0978808e-01


.. parsed-literal::

      28	 7.3015640e-01	 2.1151440e-01	 7.7078942e-01	 2.1741075e-01	[ 7.7775171e-01]	 2.0210028e-01
      29	 7.5576717e-01	 2.1351646e-01	 7.9605314e-01	 2.1996490e-01	[ 8.0156104e-01]	 1.8313527e-01


.. parsed-literal::

      30	 7.8010426e-01	 2.1434025e-01	 8.2243948e-01	 2.2009717e-01	[ 8.2621812e-01]	 2.0644164e-01


.. parsed-literal::

      31	 8.0007858e-01	 2.1570199e-01	 8.4270190e-01	 2.2183815e-01	[ 8.4835697e-01]	 2.1355414e-01
      32	 8.1928980e-01	 2.1671349e-01	 8.6264009e-01	 2.2321530e-01	[ 8.6685220e-01]	 1.9955301e-01


.. parsed-literal::

      33	 8.5654811e-01	 2.1920601e-01	 9.0128637e-01	 2.2622373e-01	[ 9.0159257e-01]	 1.9934916e-01


.. parsed-literal::

      34	 8.8027414e-01	 2.2289559e-01	 9.2782467e-01	 2.2960862e-01	[ 9.1255268e-01]	 2.0987153e-01


.. parsed-literal::

      35	 9.0333721e-01	 2.2080926e-01	 9.4987789e-01	 2.2669385e-01	[ 9.2902606e-01]	 2.1078634e-01
      36	 9.1734577e-01	 2.1576949e-01	 9.6422712e-01	 2.2146127e-01	[ 9.4826935e-01]	 1.8551922e-01


.. parsed-literal::

      37	 9.2829498e-01	 2.1438525e-01	 9.7456036e-01	 2.1975472e-01	[ 9.6085464e-01]	 1.8191886e-01


.. parsed-literal::

      38	 9.3875506e-01	 2.1348124e-01	 9.8546214e-01	 2.1873291e-01	[ 9.6885610e-01]	 2.0805550e-01


.. parsed-literal::

      39	 9.5299870e-01	 2.1156963e-01	 1.0011891e+00	 2.1632717e-01	[ 9.7627072e-01]	 2.1516085e-01
      40	 9.6519116e-01	 2.0929590e-01	 1.0142068e+00	 2.1423366e-01	[ 9.8771609e-01]	 1.8424749e-01


.. parsed-literal::

      41	 9.7723960e-01	 2.0559948e-01	 1.0265185e+00	 2.1055665e-01	[ 9.9839170e-01]	 2.0850492e-01


.. parsed-literal::

      42	 9.8657010e-01	 2.0411385e-01	 1.0357031e+00	 2.0912044e-01	[ 1.0095496e+00]	 2.1029830e-01
      43	 1.0004658e+00	 2.0210750e-01	 1.0503858e+00	 2.0739664e-01	[ 1.0280596e+00]	 2.0889115e-01


.. parsed-literal::

      44	 1.0078335e+00	 2.0271082e-01	 1.0575992e+00	 2.0738803e-01	[ 1.0359088e+00]	 2.0320439e-01


.. parsed-literal::

      45	 1.0164039e+00	 2.0183431e-01	 1.0663499e+00	 2.0662811e-01	[ 1.0431946e+00]	 2.0282626e-01


.. parsed-literal::

      46	 1.0281633e+00	 2.0088598e-01	 1.0785919e+00	 2.0584261e-01	[ 1.0518817e+00]	 2.0435476e-01


.. parsed-literal::

      47	 1.0405928e+00	 2.0048935e-01	 1.0913827e+00	 2.0632490e-01	[ 1.0598080e+00]	 2.0219636e-01
      48	 1.0482492e+00	 1.9977657e-01	 1.0999801e+00	 2.0529292e-01	[ 1.0664845e+00]	 2.0488167e-01


.. parsed-literal::

      49	 1.0570803e+00	 2.0006062e-01	 1.1084465e+00	 2.0618914e-01	[ 1.0738599e+00]	 2.0748425e-01


.. parsed-literal::

      50	 1.0619485e+00	 1.9963778e-01	 1.1133100e+00	 2.0592476e-01	[ 1.0782313e+00]	 2.1045423e-01
      51	 1.0735283e+00	 1.9867430e-01	 1.1252046e+00	 2.0504382e-01	[ 1.0889503e+00]	 2.0576143e-01


.. parsed-literal::

      52	 1.0865274e+00	 1.9697550e-01	 1.1378750e+00	 2.0322313e-01	[ 1.1005193e+00]	 2.1118689e-01


.. parsed-literal::

      53	 1.0983739e+00	 1.9484752e-01	 1.1504458e+00	 2.0073340e-01	[ 1.1118778e+00]	 2.1051693e-01


.. parsed-literal::

      54	 1.1083239e+00	 1.9346908e-01	 1.1604998e+00	 1.9963248e-01	[ 1.1199955e+00]	 2.0664716e-01


.. parsed-literal::

      55	 1.1224385e+00	 1.9069482e-01	 1.1749452e+00	 1.9720627e-01	[ 1.1286412e+00]	 2.0675826e-01


.. parsed-literal::

      56	 1.1327167e+00	 1.8955338e-01	 1.1861588e+00	 1.9614158e-01	[ 1.1296563e+00]	 2.0487547e-01


.. parsed-literal::

      57	 1.1431757e+00	 1.8703231e-01	 1.1963809e+00	 1.9365771e-01	[ 1.1370984e+00]	 2.0233011e-01


.. parsed-literal::

      58	 1.1523884e+00	 1.8734478e-01	 1.2058504e+00	 1.9379960e-01	[ 1.1426546e+00]	 2.5633407e-01


.. parsed-literal::

      59	 1.1638737e+00	 1.8439204e-01	 1.2176672e+00	 1.9060072e-01	[ 1.1520820e+00]	 2.0893288e-01
      60	 1.1789202e+00	 1.8048086e-01	 1.2327927e+00	 1.8627603e-01	[ 1.1620812e+00]	 1.9405508e-01


.. parsed-literal::

      61	 1.1918106e+00	 1.7754595e-01	 1.2463080e+00	 1.8294364e-01	[ 1.1687155e+00]	 1.8428016e-01
      62	 1.2019675e+00	 1.7643226e-01	 1.2563586e+00	 1.8181588e-01	[ 1.1756316e+00]	 2.0014453e-01


.. parsed-literal::

      63	 1.2141028e+00	 1.7315063e-01	 1.2686885e+00	 1.7831557e-01	[ 1.1801437e+00]	 1.8929839e-01
      64	 1.2242058e+00	 1.6944909e-01	 1.2788572e+00	 1.7404432e-01	[ 1.1930561e+00]	 1.8194389e-01


.. parsed-literal::

      65	 1.2345258e+00	 1.6743769e-01	 1.2891524e+00	 1.7178077e-01	[ 1.2059624e+00]	 2.1395850e-01


.. parsed-literal::

      66	 1.2485592e+00	 1.6357868e-01	 1.3040746e+00	 1.6629180e-01	[ 1.2209857e+00]	 2.1262765e-01
      67	 1.2570861e+00	 1.6261571e-01	 1.3124876e+00	 1.6535545e-01	[ 1.2316547e+00]	 1.8670511e-01


.. parsed-literal::

      68	 1.2645805e+00	 1.6284144e-01	 1.3195641e+00	 1.6553932e-01	[ 1.2393480e+00]	 1.9581485e-01


.. parsed-literal::

      69	 1.2715278e+00	 1.6184496e-01	 1.3266557e+00	 1.6392783e-01	[ 1.2439804e+00]	 2.0424414e-01


.. parsed-literal::

      70	 1.2791917e+00	 1.6125357e-01	 1.3346353e+00	 1.6323911e-01	[ 1.2451501e+00]	 2.0747900e-01
      71	 1.2901818e+00	 1.5871579e-01	 1.3456814e+00	 1.6038712e-01	[ 1.2514862e+00]	 2.0306945e-01


.. parsed-literal::

      72	 1.2997242e+00	 1.5582974e-01	 1.3556814e+00	 1.5757351e-01	[ 1.2534647e+00]	 2.0443225e-01
      73	 1.3083731e+00	 1.5484013e-01	 1.3641971e+00	 1.5683704e-01	[ 1.2605803e+00]	 1.9978857e-01


.. parsed-literal::

      74	 1.3157494e+00	 1.5386279e-01	 1.3714215e+00	 1.5593941e-01	[ 1.2707711e+00]	 2.0666933e-01


.. parsed-literal::

      75	 1.3218822e+00	 1.5282648e-01	 1.3778050e+00	 1.5427557e-01	[ 1.2750304e+00]	 2.0455480e-01


.. parsed-literal::

      76	 1.3289868e+00	 1.5178599e-01	 1.3849566e+00	 1.5284649e-01	[ 1.2833522e+00]	 2.1052432e-01


.. parsed-literal::

      77	 1.3354040e+00	 1.5065862e-01	 1.3914892e+00	 1.5122112e-01	[ 1.2893866e+00]	 2.1194935e-01


.. parsed-literal::

      78	 1.3416241e+00	 1.4905593e-01	 1.3978577e+00	 1.4935009e-01	[ 1.2937911e+00]	 2.0835233e-01


.. parsed-literal::

      79	 1.3500474e+00	 1.4653463e-01	 1.4065496e+00	 1.4691810e-01	[ 1.2967448e+00]	 2.0482492e-01


.. parsed-literal::

      80	 1.3561404e+00	 1.4526513e-01	 1.4127554e+00	 1.4576057e-01	[ 1.2986197e+00]	 2.0664382e-01
      81	 1.3634008e+00	 1.4376833e-01	 1.4202565e+00	 1.4450579e-01	  1.2975711e+00 	 2.0235968e-01


.. parsed-literal::

      82	 1.3704564e+00	 1.4274631e-01	 1.4274298e+00	 1.4341100e-01	  1.2949527e+00 	 2.0558047e-01


.. parsed-literal::

      83	 1.3763547e+00	 1.4163777e-01	 1.4332938e+00	 1.4182088e-01	[ 1.2987368e+00]	 2.0469952e-01


.. parsed-literal::

      84	 1.3809515e+00	 1.4078292e-01	 1.4379539e+00	 1.4095174e-01	[ 1.2997315e+00]	 2.4552727e-01


.. parsed-literal::

      85	 1.3866538e+00	 1.3949302e-01	 1.4437253e+00	 1.3947121e-01	[ 1.3050581e+00]	 2.0749712e-01


.. parsed-literal::

      86	 1.3926724e+00	 1.3797060e-01	 1.4498517e+00	 1.3765469e-01	[ 1.3108667e+00]	 2.2504020e-01


.. parsed-literal::

      87	 1.3986507e+00	 1.3711892e-01	 1.4557821e+00	 1.3684837e-01	[ 1.3157685e+00]	 2.2398663e-01


.. parsed-literal::

      88	 1.4034235e+00	 1.3681317e-01	 1.4606241e+00	 1.3654503e-01	[ 1.3196553e+00]	 2.0230365e-01
      89	 1.4084716e+00	 1.3604155e-01	 1.4660321e+00	 1.3580919e-01	  1.3156514e+00 	 1.8891478e-01


.. parsed-literal::

      90	 1.4135127e+00	 1.3542863e-01	 1.4712772e+00	 1.3563858e-01	  1.3128542e+00 	 2.1164536e-01


.. parsed-literal::

      91	 1.4179554e+00	 1.3496025e-01	 1.4756307e+00	 1.3507336e-01	  1.3158246e+00 	 2.1192861e-01
      92	 1.4219842e+00	 1.3429844e-01	 1.4797160e+00	 1.3449872e-01	  1.3129686e+00 	 1.9579458e-01


.. parsed-literal::

      93	 1.4269294e+00	 1.3344413e-01	 1.4847343e+00	 1.3384701e-01	  1.3141939e+00 	 1.8347621e-01


.. parsed-literal::

      94	 1.4312551e+00	 1.3290877e-01	 1.4890654e+00	 1.3372499e-01	  1.3061993e+00 	 2.0570183e-01
      95	 1.4348138e+00	 1.3231936e-01	 1.4926384e+00	 1.3342205e-01	  1.3060392e+00 	 2.0054650e-01


.. parsed-literal::

      96	 1.4376271e+00	 1.3202035e-01	 1.4954211e+00	 1.3335711e-01	  1.3057492e+00 	 2.0572710e-01


.. parsed-literal::

      97	 1.4415290e+00	 1.3154769e-01	 1.4994685e+00	 1.3338487e-01	  1.3011306e+00 	 2.0692992e-01


.. parsed-literal::

      98	 1.4449354e+00	 1.3075328e-01	 1.5032011e+00	 1.3338562e-01	  1.2908273e+00 	 2.0195436e-01


.. parsed-literal::

      99	 1.4490928e+00	 1.3063308e-01	 1.5071918e+00	 1.3327304e-01	  1.2977383e+00 	 2.0530295e-01


.. parsed-literal::

     100	 1.4525382e+00	 1.3039423e-01	 1.5106153e+00	 1.3309472e-01	  1.3010220e+00 	 2.0237637e-01
     101	 1.4566729e+00	 1.2995222e-01	 1.5148355e+00	 1.3278485e-01	  1.3002370e+00 	 1.8394232e-01


.. parsed-literal::

     102	 1.4605890e+00	 1.2926814e-01	 1.5191586e+00	 1.3309271e-01	  1.2862366e+00 	 2.0828319e-01
     103	 1.4658218e+00	 1.2869147e-01	 1.5243197e+00	 1.3252525e-01	  1.2842079e+00 	 1.9994164e-01


.. parsed-literal::

     104	 1.4687282e+00	 1.2830613e-01	 1.5272231e+00	 1.3230565e-01	  1.2826933e+00 	 2.0843029e-01


.. parsed-literal::

     105	 1.4719094e+00	 1.2774863e-01	 1.5305755e+00	 1.3222168e-01	  1.2775768e+00 	 2.0889449e-01


.. parsed-literal::

     106	 1.4745512e+00	 1.2714640e-01	 1.5334193e+00	 1.3190780e-01	  1.2749577e+00 	 2.0959592e-01
     107	 1.4772817e+00	 1.2702419e-01	 1.5360515e+00	 1.3177959e-01	  1.2834631e+00 	 2.0868659e-01


.. parsed-literal::

     108	 1.4803225e+00	 1.2676563e-01	 1.5391085e+00	 1.3155505e-01	  1.2900383e+00 	 2.0627308e-01


.. parsed-literal::

     109	 1.4827533e+00	 1.2646108e-01	 1.5415261e+00	 1.3131018e-01	  1.2956294e+00 	 2.0794916e-01
     110	 1.4865640e+00	 1.2594316e-01	 1.5453247e+00	 1.3093864e-01	  1.2951004e+00 	 2.0570040e-01


.. parsed-literal::

     111	 1.4893196e+00	 1.2561747e-01	 1.5481835e+00	 1.3090156e-01	  1.2955962e+00 	 2.0949006e-01
     112	 1.4921108e+00	 1.2545929e-01	 1.5509501e+00	 1.3090433e-01	  1.2947388e+00 	 1.9944382e-01


.. parsed-literal::

     113	 1.4951099e+00	 1.2529969e-01	 1.5540003e+00	 1.3092111e-01	  1.2883090e+00 	 2.0295453e-01
     114	 1.4977080e+00	 1.2511650e-01	 1.5566815e+00	 1.3099166e-01	  1.2821623e+00 	 1.9951415e-01


.. parsed-literal::

     115	 1.5006543e+00	 1.2495345e-01	 1.5596112e+00	 1.3085582e-01	  1.2827926e+00 	 2.0889115e-01


.. parsed-literal::

     116	 1.5039984e+00	 1.2477342e-01	 1.5629169e+00	 1.3085940e-01	  1.2714544e+00 	 2.1005106e-01


.. parsed-literal::

     117	 1.5063097e+00	 1.2480064e-01	 1.5651765e+00	 1.3067864e-01	  1.2792949e+00 	 2.1603227e-01
     118	 1.5078571e+00	 1.2477755e-01	 1.5666457e+00	 1.3056829e-01	  1.2800155e+00 	 1.9473696e-01


.. parsed-literal::

     119	 1.5099844e+00	 1.2468909e-01	 1.5688236e+00	 1.3037401e-01	  1.2707349e+00 	 2.1674204e-01
     120	 1.5116955e+00	 1.2480210e-01	 1.5706184e+00	 1.3036480e-01	  1.2694400e+00 	 2.0355201e-01


.. parsed-literal::

     121	 1.5131726e+00	 1.2483300e-01	 1.5721276e+00	 1.3027335e-01	  1.2684627e+00 	 1.8577075e-01


.. parsed-literal::

     122	 1.5159824e+00	 1.2484890e-01	 1.5750863e+00	 1.3003303e-01	  1.2677949e+00 	 2.1386814e-01
     123	 1.5175432e+00	 1.2494339e-01	 1.5767632e+00	 1.2998206e-01	  1.2589633e+00 	 2.0303893e-01


.. parsed-literal::

     124	 1.5195656e+00	 1.2481313e-01	 1.5787259e+00	 1.2990884e-01	  1.2615981e+00 	 2.0454741e-01


.. parsed-literal::

     125	 1.5215244e+00	 1.2465569e-01	 1.5807028e+00	 1.2985459e-01	  1.2602188e+00 	 2.1741867e-01
     126	 1.5228169e+00	 1.2456125e-01	 1.5820258e+00	 1.2985046e-01	  1.2577567e+00 	 2.0468903e-01


.. parsed-literal::

     127	 1.5251965e+00	 1.2442480e-01	 1.5845415e+00	 1.2980309e-01	  1.2497460e+00 	 1.9881773e-01


.. parsed-literal::

     128	 1.5268907e+00	 1.2427779e-01	 1.5863813e+00	 1.2972820e-01	  1.2424638e+00 	 2.0798993e-01
     129	 1.5284282e+00	 1.2421983e-01	 1.5878971e+00	 1.2962413e-01	  1.2427423e+00 	 2.0032382e-01


.. parsed-literal::

     130	 1.5300742e+00	 1.2412057e-01	 1.5896434e+00	 1.2939247e-01	  1.2404016e+00 	 2.1514082e-01


.. parsed-literal::

     131	 1.5311820e+00	 1.2406485e-01	 1.5908194e+00	 1.2928201e-01	  1.2406856e+00 	 2.0940614e-01
     132	 1.5329800e+00	 1.2387638e-01	 1.5926870e+00	 1.2901768e-01	  1.2369220e+00 	 1.8197656e-01


.. parsed-literal::

     133	 1.5345039e+00	 1.2369895e-01	 1.5943072e+00	 1.2855506e-01	  1.2386976e+00 	 2.0239186e-01
     134	 1.5364770e+00	 1.2362228e-01	 1.5961306e+00	 1.2851788e-01	  1.2347465e+00 	 2.0082974e-01


.. parsed-literal::

     135	 1.5377301e+00	 1.2356100e-01	 1.5972753e+00	 1.2842747e-01	  1.2342385e+00 	 1.9641066e-01
     136	 1.5392420e+00	 1.2345751e-01	 1.5987487e+00	 1.2825627e-01	  1.2337678e+00 	 1.7141867e-01


.. parsed-literal::

     137	 1.5413590e+00	 1.2331227e-01	 1.6008926e+00	 1.2805731e-01	  1.2220938e+00 	 1.9912958e-01


.. parsed-literal::

     138	 1.5433898e+00	 1.2316809e-01	 1.6030072e+00	 1.2790742e-01	  1.2228743e+00 	 2.0751834e-01
     139	 1.5449093e+00	 1.2315290e-01	 1.6045517e+00	 1.2788956e-01	  1.2242483e+00 	 1.9848228e-01


.. parsed-literal::

     140	 1.5467338e+00	 1.2313509e-01	 1.6065046e+00	 1.2797240e-01	  1.2185407e+00 	 2.3132300e-01
     141	 1.5479365e+00	 1.2307316e-01	 1.6078211e+00	 1.2790797e-01	  1.2144337e+00 	 1.9159412e-01


.. parsed-literal::

     142	 1.5493767e+00	 1.2301346e-01	 1.6091817e+00	 1.2788766e-01	  1.2119816e+00 	 2.0098138e-01
     143	 1.5508900e+00	 1.2290320e-01	 1.6106443e+00	 1.2785821e-01	  1.2059198e+00 	 2.0095897e-01


.. parsed-literal::

     144	 1.5517606e+00	 1.2282260e-01	 1.6115038e+00	 1.2783407e-01	  1.2033535e+00 	 2.0089626e-01
     145	 1.5522127e+00	 1.2265085e-01	 1.6122021e+00	 1.2790150e-01	  1.1970178e+00 	 2.0346475e-01


.. parsed-literal::

     146	 1.5544408e+00	 1.2253566e-01	 1.6143019e+00	 1.2777377e-01	  1.1913658e+00 	 1.9812083e-01


.. parsed-literal::

     147	 1.5548420e+00	 1.2254801e-01	 1.6147040e+00	 1.2776616e-01	  1.1925751e+00 	 2.3505116e-01
     148	 1.5564243e+00	 1.2248582e-01	 1.6164148e+00	 1.2768702e-01	  1.1870330e+00 	 1.9530630e-01


.. parsed-literal::

     149	 1.5582136e+00	 1.2241988e-01	 1.6184068e+00	 1.2754677e-01	  1.1743300e+00 	 1.9759870e-01


.. parsed-literal::

     150	 1.5600665e+00	 1.2233288e-01	 1.6203656e+00	 1.2742176e-01	  1.1664712e+00 	 2.1199632e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.08 s, total: 2min 5s
    Wall time: 31.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f95f8fe3cd0>



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
    CPU times: user 982 ms, sys: 41.9 ms, total: 1.02 s
    Wall time: 380 ms


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

