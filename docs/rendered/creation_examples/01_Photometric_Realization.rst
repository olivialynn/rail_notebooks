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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f77fcd05540>



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
    0      23.994413  0.000482  0.000284  
    1      25.391064  0.000343  0.000252  
    2      24.304707  0.065475  0.058724  
    3      25.291103  0.052486  0.051540  
    4      25.096743  0.008250  0.004654  
    ...          ...       ...       ...  
    99995  24.737946  0.046645  0.027453  
    99996  24.224169  0.218624  0.131991  
    99997  25.613836  0.030689  0.030423  
    99998  25.274899  0.013475  0.009051  
    99999  25.699642  0.116471  0.104930  
    
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
          <td>27.350125</td>
          <td>0.701220</td>
          <td>26.714877</td>
          <td>0.166489</td>
          <td>25.940534</td>
          <td>0.074812</td>
          <td>25.196194</td>
          <td>0.063183</td>
          <td>24.729916</td>
          <td>0.080002</td>
          <td>24.111069</td>
          <td>0.104174</td>
          <td>0.000482</td>
          <td>0.000284</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.086925</td>
          <td>0.227659</td>
          <td>26.635979</td>
          <td>0.137471</td>
          <td>26.427549</td>
          <td>0.184638</td>
          <td>25.686068</td>
          <td>0.183305</td>
          <td>25.702530</td>
          <td>0.392160</td>
          <td>0.000343</td>
          <td>0.000252</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.518276</td>
          <td>2.213637</td>
          <td>30.202191</td>
          <td>1.770103</td>
          <td>28.103083</td>
          <td>0.454971</td>
          <td>25.936193</td>
          <td>0.121123</td>
          <td>25.106649</td>
          <td>0.111355</td>
          <td>24.133025</td>
          <td>0.106193</td>
          <td>0.065475</td>
          <td>0.058724</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.240737</td>
          <td>1.075872</td>
          <td>27.208107</td>
          <td>0.223355</td>
          <td>26.492116</td>
          <td>0.194976</td>
          <td>25.283720</td>
          <td>0.129878</td>
          <td>25.201776</td>
          <td>0.263263</td>
          <td>0.052486</td>
          <td>0.051540</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.764411</td>
          <td>0.461353</td>
          <td>26.151286</td>
          <td>0.102306</td>
          <td>25.954823</td>
          <td>0.075762</td>
          <td>25.597663</td>
          <td>0.090086</td>
          <td>25.513432</td>
          <td>0.158267</td>
          <td>24.750582</td>
          <td>0.180803</td>
          <td>0.008250</td>
          <td>0.004654</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.477929</td>
          <td>0.135881</td>
          <td>25.483995</td>
          <td>0.049908</td>
          <td>25.091169</td>
          <td>0.057562</td>
          <td>24.712330</td>
          <td>0.078770</td>
          <td>24.506733</td>
          <td>0.146835</td>
          <td>0.046645</td>
          <td>0.027453</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.019410</td>
          <td>0.556440</td>
          <td>26.713642</td>
          <td>0.166314</td>
          <td>25.998099</td>
          <td>0.078714</td>
          <td>25.295594</td>
          <td>0.069000</td>
          <td>25.006248</td>
          <td>0.102001</td>
          <td>24.106630</td>
          <td>0.103770</td>
          <td>0.218624</td>
          <td>0.131991</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.586594</td>
          <td>0.403112</td>
          <td>26.603952</td>
          <td>0.151435</td>
          <td>26.277816</td>
          <td>0.100673</td>
          <td>26.200122</td>
          <td>0.152122</td>
          <td>26.049904</td>
          <td>0.248353</td>
          <td>25.434846</td>
          <td>0.317792</td>
          <td>0.030689</td>
          <td>0.030423</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.902378</td>
          <td>0.233323</td>
          <td>26.327297</td>
          <td>0.119268</td>
          <td>26.003844</td>
          <td>0.079115</td>
          <td>25.788712</td>
          <td>0.106514</td>
          <td>25.928751</td>
          <td>0.224683</td>
          <td>25.263327</td>
          <td>0.276800</td>
          <td>0.013475</td>
          <td>0.009051</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.755724</td>
          <td>0.206550</td>
          <td>26.924859</td>
          <td>0.198852</td>
          <td>26.436151</td>
          <td>0.115604</td>
          <td>26.061157</td>
          <td>0.134972</td>
          <td>25.462550</td>
          <td>0.151518</td>
          <td>26.432626</td>
          <td>0.669118</td>
          <td>0.116471</td>
          <td>0.104930</td>
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
          <td>28.511211</td>
          <td>1.502023</td>
          <td>26.777336</td>
          <td>0.201397</td>
          <td>26.033412</td>
          <td>0.095481</td>
          <td>25.192648</td>
          <td>0.074654</td>
          <td>24.687020</td>
          <td>0.090580</td>
          <td>24.413578</td>
          <td>0.159544</td>
          <td>0.000482</td>
          <td>0.000284</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.610182</td>
          <td>0.906069</td>
          <td>28.841357</td>
          <td>0.933066</td>
          <td>26.621855</td>
          <td>0.159049</td>
          <td>26.294741</td>
          <td>0.194032</td>
          <td>25.420116</td>
          <td>0.170944</td>
          <td>27.126605</td>
          <td>1.164423</td>
          <td>0.000343</td>
          <td>0.000252</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.991463</td>
          <td>0.604735</td>
          <td>28.841085</td>
          <td>0.940620</td>
          <td>28.294740</td>
          <td>0.604762</td>
          <td>25.846136</td>
          <td>0.134044</td>
          <td>25.091096</td>
          <td>0.130544</td>
          <td>24.311045</td>
          <td>0.148045</td>
          <td>0.065475</td>
          <td>0.058724</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.626514</td>
          <td>0.819413</td>
          <td>27.161910</td>
          <td>0.252400</td>
          <td>26.551811</td>
          <td>0.242560</td>
          <td>25.373167</td>
          <td>0.165706</td>
          <td>25.377540</td>
          <td>0.355934</td>
          <td>0.052486</td>
          <td>0.051540</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.081309</td>
          <td>0.301027</td>
          <td>26.063223</td>
          <td>0.109232</td>
          <td>25.942575</td>
          <td>0.088171</td>
          <td>25.631229</td>
          <td>0.109770</td>
          <td>25.056970</td>
          <td>0.125147</td>
          <td>25.072062</td>
          <td>0.276530</td>
          <td>0.008250</td>
          <td>0.004654</td>
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
          <td>26.731939</td>
          <td>0.498926</td>
          <td>26.428295</td>
          <td>0.150427</td>
          <td>25.370656</td>
          <td>0.053424</td>
          <td>25.011020</td>
          <td>0.063901</td>
          <td>24.898419</td>
          <td>0.109544</td>
          <td>24.436611</td>
          <td>0.163521</td>
          <td>0.046645</td>
          <td>0.027453</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.910231</td>
          <td>0.602424</td>
          <td>27.026938</td>
          <td>0.269310</td>
          <td>26.106905</td>
          <td>0.112274</td>
          <td>25.266048</td>
          <td>0.088171</td>
          <td>24.706177</td>
          <td>0.101597</td>
          <td>24.182997</td>
          <td>0.144426</td>
          <td>0.218624</td>
          <td>0.131991</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.223124</td>
          <td>0.337712</td>
          <td>26.978734</td>
          <td>0.238791</td>
          <td>26.502687</td>
          <td>0.144043</td>
          <td>26.248323</td>
          <td>0.187173</td>
          <td>26.148343</td>
          <td>0.313106</td>
          <td>25.576010</td>
          <td>0.412907</td>
          <td>0.030689</td>
          <td>0.030423</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.777080</td>
          <td>0.235025</td>
          <td>26.366946</td>
          <td>0.142141</td>
          <td>26.095788</td>
          <td>0.100892</td>
          <td>25.984287</td>
          <td>0.149068</td>
          <td>25.713276</td>
          <td>0.218896</td>
          <td>25.047594</td>
          <td>0.271159</td>
          <td>0.013475</td>
          <td>0.009051</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.834158</td>
          <td>1.780618</td>
          <td>26.689953</td>
          <td>0.193759</td>
          <td>26.757782</td>
          <td>0.185600</td>
          <td>26.588966</td>
          <td>0.257685</td>
          <td>26.148016</td>
          <td>0.323693</td>
          <td>28.074244</td>
          <td>1.909420</td>
          <td>0.116471</td>
          <td>0.104930</td>
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
          <td>26.859796</td>
          <td>0.495288</td>
          <td>26.597450</td>
          <td>0.150593</td>
          <td>26.172901</td>
          <td>0.091819</td>
          <td>25.162892</td>
          <td>0.061344</td>
          <td>24.612940</td>
          <td>0.072147</td>
          <td>23.979581</td>
          <td>0.092833</td>
          <td>0.000482</td>
          <td>0.000284</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.363454</td>
          <td>0.610780</td>
          <td>26.892675</td>
          <td>0.171294</td>
          <td>26.157648</td>
          <td>0.146675</td>
          <td>25.631010</td>
          <td>0.174947</td>
          <td>25.693148</td>
          <td>0.389327</td>
          <td>0.000343</td>
          <td>0.000252</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.533623</td>
          <td>0.399079</td>
          <td>29.090985</td>
          <td>1.013914</td>
          <td>27.698370</td>
          <td>0.347990</td>
          <td>25.954379</td>
          <td>0.129562</td>
          <td>25.180061</td>
          <td>0.124700</td>
          <td>24.263995</td>
          <td>0.125283</td>
          <td>0.065475</td>
          <td>0.058724</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.054827</td>
          <td>0.582377</td>
          <td>29.265282</td>
          <td>1.113737</td>
          <td>27.559999</td>
          <td>0.307723</td>
          <td>26.301717</td>
          <td>0.172010</td>
          <td>25.602119</td>
          <td>0.176647</td>
          <td>25.738620</td>
          <td>0.416366</td>
          <td>0.052486</td>
          <td>0.051540</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.179871</td>
          <td>0.292744</td>
          <td>26.304434</td>
          <td>0.116983</td>
          <td>25.837305</td>
          <td>0.068323</td>
          <td>25.893753</td>
          <td>0.116808</td>
          <td>25.572772</td>
          <td>0.166589</td>
          <td>25.095027</td>
          <td>0.241314</td>
          <td>0.008250</td>
          <td>0.004654</td>
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
          <td>26.975144</td>
          <td>0.544969</td>
          <td>26.384800</td>
          <td>0.127475</td>
          <td>25.441623</td>
          <td>0.049020</td>
          <td>25.082303</td>
          <td>0.058301</td>
          <td>24.919508</td>
          <td>0.096384</td>
          <td>24.594724</td>
          <td>0.161478</td>
          <td>0.046645</td>
          <td>0.027453</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.086142</td>
          <td>0.292922</td>
          <td>26.394407</td>
          <td>0.150010</td>
          <td>25.169650</td>
          <td>0.084531</td>
          <td>24.757471</td>
          <td>0.110750</td>
          <td>24.082346</td>
          <td>0.138053</td>
          <td>0.218624</td>
          <td>0.131991</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.771575</td>
          <td>0.927664</td>
          <td>26.972875</td>
          <td>0.209176</td>
          <td>26.297845</td>
          <td>0.103741</td>
          <td>26.632575</td>
          <td>0.222062</td>
          <td>26.143432</td>
          <td>0.271249</td>
          <td>25.556901</td>
          <td>0.354161</td>
          <td>0.030689</td>
          <td>0.030423</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.990973</td>
          <td>0.251266</td>
          <td>26.220369</td>
          <td>0.108836</td>
          <td>26.129374</td>
          <td>0.088529</td>
          <td>25.836760</td>
          <td>0.111285</td>
          <td>25.751901</td>
          <td>0.194112</td>
          <td>26.117288</td>
          <td>0.536156</td>
          <td>0.013475</td>
          <td>0.009051</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.224627</td>
          <td>0.697483</td>
          <td>27.184870</td>
          <td>0.277612</td>
          <td>26.553261</td>
          <td>0.147295</td>
          <td>26.114098</td>
          <td>0.163439</td>
          <td>26.242657</td>
          <td>0.330991</td>
          <td>25.813023</td>
          <td>0.484413</td>
          <td>0.116471</td>
          <td>0.104930</td>
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
