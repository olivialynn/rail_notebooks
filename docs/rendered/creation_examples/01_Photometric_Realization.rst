Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f21906d3dc0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>27.873360</td>
          <td>0.981372</td>
          <td>26.483326</td>
          <td>0.136515</td>
          <td>26.092087</td>
          <td>0.085517</td>
          <td>25.265837</td>
          <td>0.067206</td>
          <td>24.738076</td>
          <td>0.080580</td>
          <td>24.006742</td>
          <td>0.095073</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.103807</td>
          <td>0.230867</td>
          <td>26.649225</td>
          <td>0.139051</td>
          <td>26.466474</td>
          <td>0.190808</td>
          <td>25.634171</td>
          <td>0.175417</td>
          <td>25.102584</td>
          <td>0.242678</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.979696</td>
          <td>0.919232</td>
          <td>28.932140</td>
          <td>0.814169</td>
          <td>26.160531</td>
          <td>0.147039</td>
          <td>24.948720</td>
          <td>0.096987</td>
          <td>24.223573</td>
          <td>0.114922</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.173068</td>
          <td>0.216933</td>
          <td>26.097381</td>
          <td>0.139258</td>
          <td>25.531759</td>
          <td>0.160766</td>
          <td>25.300786</td>
          <td>0.285335</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.823581</td>
          <td>0.218571</td>
          <td>26.091632</td>
          <td>0.097102</td>
          <td>25.985680</td>
          <td>0.077856</td>
          <td>25.706829</td>
          <td>0.099148</td>
          <td>25.849404</td>
          <td>0.210303</td>
          <td>24.796968</td>
          <td>0.188036</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>28.621527</td>
          <td>1.484809</td>
          <td>26.343670</td>
          <td>0.120976</td>
          <td>25.409763</td>
          <td>0.046725</td>
          <td>25.053867</td>
          <td>0.055687</td>
          <td>24.916697</td>
          <td>0.094299</td>
          <td>24.610219</td>
          <td>0.160452</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.456431</td>
          <td>0.753061</td>
          <td>26.738216</td>
          <td>0.169829</td>
          <td>26.113177</td>
          <td>0.087120</td>
          <td>25.188632</td>
          <td>0.062761</td>
          <td>24.850859</td>
          <td>0.088998</td>
          <td>24.227071</td>
          <td>0.115273</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.938910</td>
          <td>0.524905</td>
          <td>27.027471</td>
          <td>0.216679</td>
          <td>26.451916</td>
          <td>0.117201</td>
          <td>26.306557</td>
          <td>0.166614</td>
          <td>25.936363</td>
          <td>0.226108</td>
          <td>25.560827</td>
          <td>0.351146</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.677564</td>
          <td>0.432107</td>
          <td>26.223175</td>
          <td>0.108935</td>
          <td>26.015752</td>
          <td>0.079950</td>
          <td>25.920454</td>
          <td>0.119477</td>
          <td>25.551273</td>
          <td>0.163467</td>
          <td>26.353327</td>
          <td>0.633364</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.199094</td>
          <td>0.632027</td>
          <td>26.809913</td>
          <td>0.180482</td>
          <td>26.598385</td>
          <td>0.133079</td>
          <td>26.103802</td>
          <td>0.140032</td>
          <td>25.945798</td>
          <td>0.227886</td>
          <td>25.611515</td>
          <td>0.365381</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>29.734717</td>
          <td>2.523770</td>
          <td>26.839517</td>
          <td>0.212160</td>
          <td>26.186634</td>
          <td>0.109185</td>
          <td>25.111671</td>
          <td>0.069498</td>
          <td>24.905944</td>
          <td>0.109728</td>
          <td>24.172005</td>
          <td>0.129614</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.393002</td>
          <td>0.333112</td>
          <td>26.372755</td>
          <td>0.128390</td>
          <td>26.127927</td>
          <td>0.168511</td>
          <td>25.923055</td>
          <td>0.260239</td>
          <td>25.599032</td>
          <td>0.419139</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.596931</td>
          <td>1.581720</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.326387</td>
          <td>0.623003</td>
          <td>26.105333</td>
          <td>0.169003</td>
          <td>25.250546</td>
          <td>0.151140</td>
          <td>24.414398</td>
          <td>0.163256</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.686405</td>
          <td>0.884516</td>
          <td>27.129083</td>
          <td>0.259363</td>
          <td>26.594474</td>
          <td>0.265651</td>
          <td>25.299861</td>
          <td>0.164680</td>
          <td>24.571215</td>
          <td>0.194919</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.228739</td>
          <td>0.338560</td>
          <td>26.267168</td>
          <td>0.130412</td>
          <td>25.826059</td>
          <td>0.079586</td>
          <td>25.637893</td>
          <td>0.110433</td>
          <td>25.433918</td>
          <td>0.173021</td>
          <td>25.236656</td>
          <td>0.315793</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>29.539777</td>
          <td>2.365673</td>
          <td>26.219956</td>
          <td>0.127521</td>
          <td>25.393959</td>
          <td>0.055435</td>
          <td>25.078242</td>
          <td>0.068968</td>
          <td>24.807028</td>
          <td>0.102767</td>
          <td>24.583469</td>
          <td>0.188203</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.223594</td>
          <td>0.338076</td>
          <td>26.564365</td>
          <td>0.168849</td>
          <td>26.042252</td>
          <td>0.096630</td>
          <td>25.341788</td>
          <td>0.085525</td>
          <td>24.874743</td>
          <td>0.107223</td>
          <td>24.107259</td>
          <td>0.123061</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.939440</td>
          <td>0.233064</td>
          <td>26.512153</td>
          <td>0.146598</td>
          <td>26.816226</td>
          <td>0.301887</td>
          <td>26.076282</td>
          <td>0.298133</td>
          <td>25.446764</td>
          <td>0.376976</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.255983</td>
          <td>0.733414</td>
          <td>26.072821</td>
          <td>0.113288</td>
          <td>25.963372</td>
          <td>0.092657</td>
          <td>25.862248</td>
          <td>0.138492</td>
          <td>25.723116</td>
          <td>0.227257</td>
          <td>26.165042</td>
          <td>0.650423</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.425303</td>
          <td>0.397254</td>
          <td>26.700514</td>
          <td>0.190450</td>
          <td>26.544002</td>
          <td>0.150258</td>
          <td>26.318767</td>
          <td>0.199976</td>
          <td>25.716441</td>
          <td>0.221481</td>
          <td>25.492540</td>
          <td>0.389627</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>26.206604</td>
          <td>0.299016</td>
          <td>27.156284</td>
          <td>0.241126</td>
          <td>25.940712</td>
          <td>0.074833</td>
          <td>25.340017</td>
          <td>0.071778</td>
          <td>24.665379</td>
          <td>0.075580</td>
          <td>23.976132</td>
          <td>0.092564</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.753032</td>
          <td>0.389027</td>
          <td>26.944716</td>
          <td>0.179196</td>
          <td>26.143679</td>
          <td>0.145065</td>
          <td>26.014076</td>
          <td>0.241345</td>
          <td>25.276707</td>
          <td>0.280077</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.646734</td>
          <td>1.406948</td>
          <td>27.783638</td>
          <td>0.382705</td>
          <td>26.202361</td>
          <td>0.165643</td>
          <td>24.969533</td>
          <td>0.107160</td>
          <td>24.258306</td>
          <td>0.128795</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.352269</td>
          <td>1.301348</td>
          <td>28.140748</td>
          <td>0.564625</td>
          <td>26.292420</td>
          <td>0.206205</td>
          <td>25.346497</td>
          <td>0.170759</td>
          <td>25.570090</td>
          <td>0.433747</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.236241</td>
          <td>0.306458</td>
          <td>26.066019</td>
          <td>0.095064</td>
          <td>25.771676</td>
          <td>0.064517</td>
          <td>25.860172</td>
          <td>0.113537</td>
          <td>25.256985</td>
          <td>0.127083</td>
          <td>24.941038</td>
          <td>0.212523</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.363462</td>
          <td>0.356164</td>
          <td>26.217223</td>
          <td>0.116050</td>
          <td>25.481357</td>
          <td>0.053925</td>
          <td>25.060531</td>
          <td>0.060887</td>
          <td>25.026363</td>
          <td>0.112258</td>
          <td>24.725903</td>
          <td>0.191521</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.858176</td>
          <td>0.980105</td>
          <td>26.599340</td>
          <td>0.152960</td>
          <td>26.046004</td>
          <td>0.083483</td>
          <td>25.275344</td>
          <td>0.068967</td>
          <td>24.984018</td>
          <td>0.101691</td>
          <td>24.093999</td>
          <td>0.104395</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.199892</td>
          <td>0.306890</td>
          <td>26.809652</td>
          <td>0.188013</td>
          <td>26.134773</td>
          <td>0.093250</td>
          <td>26.312546</td>
          <td>0.175979</td>
          <td>26.056458</td>
          <td>0.261337</td>
          <td>25.150522</td>
          <td>0.264657</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.793664</td>
          <td>0.505382</td>
          <td>26.255774</td>
          <td>0.123850</td>
          <td>25.998826</td>
          <td>0.088379</td>
          <td>25.850861</td>
          <td>0.126608</td>
          <td>26.263562</td>
          <td>0.328010</td>
          <td>25.464471</td>
          <td>0.361949</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.253674</td>
          <td>0.670600</td>
          <td>26.775889</td>
          <td>0.181183</td>
          <td>26.569033</td>
          <td>0.134809</td>
          <td>26.254502</td>
          <td>0.165816</td>
          <td>26.126215</td>
          <td>0.274081</td>
          <td>25.834537</td>
          <td>0.449216</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
