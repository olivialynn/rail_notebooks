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

    <pzflow.flow.Flow at 0x7f79ffbd46d0>



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
    0      23.994413  0.143030  0.139231  
    1      25.391064  0.016947  0.010473  
    2      24.304707  0.041710  0.023175  
    3      25.291103  0.055033  0.033402  
    4      25.096743  0.082814  0.069767  
    ...          ...       ...       ...  
    99995  24.737946  0.016914  0.016663  
    99996  24.224169  0.003774  0.002739  
    99997  25.613836  0.071013  0.063777  
    99998  25.274899  0.116841  0.080514  
    99999  25.699642  0.012825  0.011064  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>26.400276</td>
          <td>0.348757</td>
          <td>26.890531</td>
          <td>0.193194</td>
          <td>26.071009</td>
          <td>0.083943</td>
          <td>25.283047</td>
          <td>0.068238</td>
          <td>24.723032</td>
          <td>0.079518</td>
          <td>23.905272</td>
          <td>0.086960</td>
          <td>0.143030</td>
          <td>0.139231</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.603448</td>
          <td>1.471367</td>
          <td>27.178997</td>
          <td>0.245654</td>
          <td>26.785435</td>
          <td>0.156315</td>
          <td>26.298276</td>
          <td>0.165442</td>
          <td>25.540294</td>
          <td>0.161942</td>
          <td>26.537463</td>
          <td>0.718595</td>
          <td>0.016947</td>
          <td>0.010473</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.408156</td>
          <td>2.118510</td>
          <td>28.843109</td>
          <td>0.843355</td>
          <td>29.779882</td>
          <td>1.340716</td>
          <td>26.036770</td>
          <td>0.132157</td>
          <td>24.988449</td>
          <td>0.100424</td>
          <td>24.190480</td>
          <td>0.111655</td>
          <td>0.041710</td>
          <td>0.023175</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.766109</td>
          <td>0.461941</td>
          <td>27.947684</td>
          <td>0.451034</td>
          <td>27.803390</td>
          <td>0.361446</td>
          <td>26.371941</td>
          <td>0.176142</td>
          <td>25.463529</td>
          <td>0.151645</td>
          <td>24.815624</td>
          <td>0.191020</td>
          <td>0.055033</td>
          <td>0.033402</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.242544</td>
          <td>0.307733</td>
          <td>25.949510</td>
          <td>0.085713</td>
          <td>25.853578</td>
          <td>0.069272</td>
          <td>25.626917</td>
          <td>0.092432</td>
          <td>25.638685</td>
          <td>0.176091</td>
          <td>24.996322</td>
          <td>0.222235</td>
          <td>0.082814</td>
          <td>0.069767</td>
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
          <td>27.073237</td>
          <td>0.578325</td>
          <td>26.447357</td>
          <td>0.132341</td>
          <td>25.493503</td>
          <td>0.050331</td>
          <td>25.092848</td>
          <td>0.057648</td>
          <td>24.795625</td>
          <td>0.084774</td>
          <td>25.085383</td>
          <td>0.239259</td>
          <td>0.016914</td>
          <td>0.016663</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.704592</td>
          <td>0.165037</td>
          <td>26.027501</td>
          <td>0.080784</td>
          <td>25.303387</td>
          <td>0.069478</td>
          <td>24.833856</td>
          <td>0.087676</td>
          <td>24.132855</td>
          <td>0.106177</td>
          <td>0.003774</td>
          <td>0.002739</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.180378</td>
          <td>1.174213</td>
          <td>26.616720</td>
          <td>0.153100</td>
          <td>26.412354</td>
          <td>0.113232</td>
          <td>26.059356</td>
          <td>0.134762</td>
          <td>26.072903</td>
          <td>0.253090</td>
          <td>25.376832</td>
          <td>0.303373</td>
          <td>0.071013</td>
          <td>0.063777</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.058893</td>
          <td>0.265305</td>
          <td>26.206653</td>
          <td>0.107376</td>
          <td>26.187785</td>
          <td>0.093028</td>
          <td>25.810743</td>
          <td>0.108584</td>
          <td>25.591996</td>
          <td>0.169240</td>
          <td>25.241999</td>
          <td>0.272041</td>
          <td>0.116841</td>
          <td>0.080514</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.556190</td>
          <td>0.393788</td>
          <td>27.019848</td>
          <td>0.215306</td>
          <td>26.395282</td>
          <td>0.111559</td>
          <td>26.301743</td>
          <td>0.165932</td>
          <td>26.246119</td>
          <td>0.291414</td>
          <td>26.340133</td>
          <td>0.627553</td>
          <td>0.012825</td>
          <td>0.011064</td>
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
          <td>26.863528</td>
          <td>0.569892</td>
          <td>26.636519</td>
          <td>0.189049</td>
          <td>25.919531</td>
          <td>0.092065</td>
          <td>25.157792</td>
          <td>0.077336</td>
          <td>24.753671</td>
          <td>0.102319</td>
          <td>24.001147</td>
          <td>0.119189</td>
          <td>0.143030</td>
          <td>0.139231</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.009671</td>
          <td>0.532875</td>
          <td>26.556230</td>
          <td>0.150456</td>
          <td>26.291619</td>
          <td>0.193652</td>
          <td>26.426304</td>
          <td>0.388783</td>
          <td>25.097850</td>
          <td>0.282516</td>
          <td>0.016947</td>
          <td>0.010473</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.787943</td>
          <td>0.837511</td>
          <td>25.945678</td>
          <td>0.144703</td>
          <td>24.873759</td>
          <td>0.107094</td>
          <td>24.395244</td>
          <td>0.157671</td>
          <td>0.041710</td>
          <td>0.023175</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.743607</td>
          <td>0.881786</td>
          <td>27.393551</td>
          <td>0.304028</td>
          <td>25.881891</td>
          <td>0.137404</td>
          <td>25.460265</td>
          <td>0.178072</td>
          <td>24.970352</td>
          <td>0.256177</td>
          <td>0.055033</td>
          <td>0.033402</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.917993</td>
          <td>0.267425</td>
          <td>26.511513</td>
          <td>0.163607</td>
          <td>25.832233</td>
          <td>0.081581</td>
          <td>25.917363</td>
          <td>0.143487</td>
          <td>25.541475</td>
          <td>0.193040</td>
          <td>25.175479</td>
          <td>0.306158</td>
          <td>0.082814</td>
          <td>0.069767</td>
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
          <td>26.771117</td>
          <td>0.512204</td>
          <td>26.378289</td>
          <td>0.143599</td>
          <td>25.493132</td>
          <td>0.059315</td>
          <td>25.088931</td>
          <td>0.068179</td>
          <td>24.852443</td>
          <td>0.104815</td>
          <td>24.892026</td>
          <td>0.238800</td>
          <td>0.016914</td>
          <td>0.016663</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.190365</td>
          <td>2.046504</td>
          <td>26.428602</td>
          <td>0.149815</td>
          <td>26.103283</td>
          <td>0.101514</td>
          <td>25.097748</td>
          <td>0.068647</td>
          <td>24.975055</td>
          <td>0.116538</td>
          <td>24.132161</td>
          <td>0.125217</td>
          <td>0.003774</td>
          <td>0.002739</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.481425</td>
          <td>0.842250</td>
          <td>27.525071</td>
          <td>0.374030</td>
          <td>26.607387</td>
          <td>0.159447</td>
          <td>25.882243</td>
          <td>0.138607</td>
          <td>25.766099</td>
          <td>0.231939</td>
          <td>25.777598</td>
          <td>0.485853</td>
          <td>0.071013</td>
          <td>0.063777</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.456769</td>
          <td>0.413275</td>
          <td>26.202867</td>
          <td>0.127003</td>
          <td>26.133642</td>
          <td>0.107725</td>
          <td>25.677945</td>
          <td>0.118249</td>
          <td>25.684543</td>
          <td>0.220412</td>
          <td>24.783351</td>
          <td>0.225121</td>
          <td>0.116841</td>
          <td>0.080514</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.211047</td>
          <td>0.333890</td>
          <td>26.473975</td>
          <td>0.155813</td>
          <td>26.126814</td>
          <td>0.103673</td>
          <td>26.756397</td>
          <td>0.284319</td>
          <td>26.134166</td>
          <td>0.308820</td>
          <td>25.625332</td>
          <td>0.427722</td>
          <td>0.012825</td>
          <td>0.011064</td>
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
          <td>26.659130</td>
          <td>0.487726</td>
          <td>26.906931</td>
          <td>0.234802</td>
          <td>26.057606</td>
          <td>0.102826</td>
          <td>25.159390</td>
          <td>0.076605</td>
          <td>24.932782</td>
          <td>0.118379</td>
          <td>24.013558</td>
          <td>0.119206</td>
          <td>0.143030</td>
          <td>0.139231</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.254018</td>
          <td>0.657548</td>
          <td>27.307957</td>
          <td>0.273581</td>
          <td>26.919080</td>
          <td>0.175636</td>
          <td>26.241551</td>
          <td>0.158051</td>
          <td>25.661420</td>
          <td>0.179984</td>
          <td>25.352706</td>
          <td>0.298308</td>
          <td>0.016947</td>
          <td>0.010473</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.865649</td>
          <td>0.788470</td>
          <td>25.939561</td>
          <td>0.123404</td>
          <td>25.035445</td>
          <td>0.106231</td>
          <td>24.296378</td>
          <td>0.124355</td>
          <td>0.041710</td>
          <td>0.023175</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.327118</td>
          <td>1.288329</td>
          <td>27.345169</td>
          <td>0.287569</td>
          <td>27.015718</td>
          <td>0.195137</td>
          <td>26.333969</td>
          <td>0.175322</td>
          <td>25.561343</td>
          <td>0.169281</td>
          <td>24.974438</td>
          <td>0.224140</td>
          <td>0.055033</td>
          <td>0.033402</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.373826</td>
          <td>0.357873</td>
          <td>26.163744</td>
          <td>0.110272</td>
          <td>26.046041</td>
          <td>0.088409</td>
          <td>25.930603</td>
          <td>0.130057</td>
          <td>25.481181</td>
          <td>0.165368</td>
          <td>25.746273</td>
          <td>0.433372</td>
          <td>0.082814</td>
          <td>0.069767</td>
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
          <td>26.754550</td>
          <td>0.458989</td>
          <td>26.255130</td>
          <td>0.112380</td>
          <td>25.392794</td>
          <td>0.046204</td>
          <td>25.169321</td>
          <td>0.061946</td>
          <td>24.933386</td>
          <td>0.096056</td>
          <td>24.842031</td>
          <td>0.196060</td>
          <td>0.016914</td>
          <td>0.016663</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.131853</td>
          <td>0.602941</td>
          <td>26.658826</td>
          <td>0.158735</td>
          <td>26.091945</td>
          <td>0.085519</td>
          <td>25.219272</td>
          <td>0.064499</td>
          <td>24.814377</td>
          <td>0.086198</td>
          <td>24.382798</td>
          <td>0.131975</td>
          <td>0.003774</td>
          <td>0.002739</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.036073</td>
          <td>0.582251</td>
          <td>27.013064</td>
          <td>0.224656</td>
          <td>26.362288</td>
          <td>0.114842</td>
          <td>26.562502</td>
          <td>0.219180</td>
          <td>25.882850</td>
          <td>0.228490</td>
          <td>25.439318</td>
          <td>0.336805</td>
          <td>0.071013</td>
          <td>0.063777</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.231221</td>
          <td>0.329580</td>
          <td>25.887563</td>
          <td>0.090160</td>
          <td>26.441386</td>
          <td>0.130605</td>
          <td>25.943791</td>
          <td>0.137782</td>
          <td>25.452781</td>
          <td>0.168651</td>
          <td>26.163967</td>
          <td>0.611741</td>
          <td>0.116841</td>
          <td>0.080514</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.470943</td>
          <td>0.369038</td>
          <td>26.943600</td>
          <td>0.202330</td>
          <td>26.584343</td>
          <td>0.131725</td>
          <td>26.296764</td>
          <td>0.165557</td>
          <td>25.853687</td>
          <td>0.211448</td>
          <td>25.005570</td>
          <td>0.224379</td>
          <td>0.012825</td>
          <td>0.011064</td>
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
