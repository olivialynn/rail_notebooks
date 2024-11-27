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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f43b8bc91e0>



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
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
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
          <td>0.890625</td>
          <td>26.332910</td>
          <td>0.330695</td>
          <td>26.717530</td>
          <td>0.166866</td>
          <td>25.853679</td>
          <td>0.069278</td>
          <td>25.358817</td>
          <td>0.072971</td>
          <td>25.190816</td>
          <td>0.119822</td>
          <td>25.297663</td>
          <td>0.284615</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.421273</td>
          <td>0.266239</td>
          <td>27.059422</td>
          <td>0.310872</td>
          <td>26.500976</td>
          <td>0.356958</td>
          <td>26.693541</td>
          <td>0.796939</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.885738</td>
          <td>0.988751</td>
          <td>25.988059</td>
          <td>0.088667</td>
          <td>24.757485</td>
          <td>0.026285</td>
          <td>23.857814</td>
          <td>0.019521</td>
          <td>23.158761</td>
          <td>0.020100</td>
          <td>22.873941</td>
          <td>0.034886</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.480943</td>
          <td>1.381824</td>
          <td>28.917716</td>
          <td>0.884276</td>
          <td>27.181029</td>
          <td>0.218378</td>
          <td>26.634817</td>
          <td>0.219723</td>
          <td>26.369802</td>
          <td>0.321797</td>
          <td>25.148405</td>
          <td>0.252002</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.961915</td>
          <td>0.245058</td>
          <td>25.793319</td>
          <td>0.074693</td>
          <td>25.431711</td>
          <td>0.047644</td>
          <td>24.785541</td>
          <td>0.043882</td>
          <td>24.397430</td>
          <td>0.059606</td>
          <td>23.755141</td>
          <td>0.076176</td>
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
          <td>2.147172</td>
          <td>26.512134</td>
          <td>0.380595</td>
          <td>26.338450</td>
          <td>0.120429</td>
          <td>26.102339</td>
          <td>0.086292</td>
          <td>26.042907</td>
          <td>0.132860</td>
          <td>25.954066</td>
          <td>0.229454</td>
          <td>25.489068</td>
          <td>0.331799</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.233957</td>
          <td>0.647537</td>
          <td>26.673098</td>
          <td>0.160662</td>
          <td>26.825265</td>
          <td>0.161730</td>
          <td>26.272722</td>
          <td>0.161873</td>
          <td>26.163055</td>
          <td>0.272442</td>
          <td>25.347218</td>
          <td>0.296234</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.327180</td>
          <td>0.277289</td>
          <td>26.805760</td>
          <td>0.159057</td>
          <td>26.792062</td>
          <td>0.250252</td>
          <td>25.784751</td>
          <td>0.199209</td>
          <td>25.131425</td>
          <td>0.248510</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.058530</td>
          <td>0.572281</td>
          <td>27.850788</td>
          <td>0.419079</td>
          <td>26.783223</td>
          <td>0.156019</td>
          <td>25.843815</td>
          <td>0.111764</td>
          <td>25.733437</td>
          <td>0.190788</td>
          <td>25.340069</td>
          <td>0.294533</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.327063</td>
          <td>0.690317</td>
          <td>26.749709</td>
          <td>0.171497</td>
          <td>26.160771</td>
          <td>0.090845</td>
          <td>25.670077</td>
          <td>0.096003</td>
          <td>25.130187</td>
          <td>0.113663</td>
          <td>24.528401</td>
          <td>0.149594</td>
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
          <td>0.890625</td>
          <td>26.823676</td>
          <td>0.531936</td>
          <td>26.650667</td>
          <td>0.181017</td>
          <td>25.792298</td>
          <td>0.077224</td>
          <td>25.352796</td>
          <td>0.085984</td>
          <td>24.942623</td>
          <td>0.113293</td>
          <td>25.016540</td>
          <td>0.264274</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.895223</td>
          <td>1.076478</td>
          <td>29.771368</td>
          <td>1.559969</td>
          <td>27.601408</td>
          <td>0.356381</td>
          <td>27.199344</td>
          <td>0.403341</td>
          <td>27.112039</td>
          <td>0.643681</td>
          <td>25.457522</td>
          <td>0.375821</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.150096</td>
          <td>0.679049</td>
          <td>25.854474</td>
          <td>0.092848</td>
          <td>24.830840</td>
          <td>0.033713</td>
          <td>23.864911</td>
          <td>0.023699</td>
          <td>23.169049</td>
          <td>0.024294</td>
          <td>22.738695</td>
          <td>0.037505</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.494119</td>
          <td>0.435086</td>
          <td>29.728724</td>
          <td>1.580140</td>
          <td>27.464704</td>
          <td>0.339784</td>
          <td>26.487243</td>
          <td>0.243286</td>
          <td>26.132991</td>
          <td>0.327760</td>
          <td>26.736445</td>
          <td>0.969692</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.919288</td>
          <td>0.264089</td>
          <td>25.761051</td>
          <td>0.083848</td>
          <td>25.444440</td>
          <td>0.056773</td>
          <td>24.782395</td>
          <td>0.051922</td>
          <td>24.439626</td>
          <td>0.072858</td>
          <td>23.640013</td>
          <td>0.081430</td>
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
          <td>2.147172</td>
          <td>26.320993</td>
          <td>0.369137</td>
          <td>26.347995</td>
          <td>0.142414</td>
          <td>26.274257</td>
          <td>0.120323</td>
          <td>26.259151</td>
          <td>0.192270</td>
          <td>25.881610</td>
          <td>0.256473</td>
          <td>25.046183</td>
          <td>0.276198</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.761980</td>
          <td>0.509831</td>
          <td>26.824926</td>
          <td>0.210342</td>
          <td>26.722428</td>
          <td>0.173988</td>
          <td>26.306426</td>
          <td>0.196767</td>
          <td>26.287134</td>
          <td>0.349868</td>
          <td>26.117882</td>
          <td>0.615553</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.960359</td>
          <td>0.591508</td>
          <td>26.799265</td>
          <td>0.207407</td>
          <td>26.823963</td>
          <td>0.191178</td>
          <td>26.501185</td>
          <td>0.233460</td>
          <td>26.033623</td>
          <td>0.288051</td>
          <td>24.562855</td>
          <td>0.183460</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.094604</td>
          <td>0.657217</td>
          <td>26.916822</td>
          <td>0.232405</td>
          <td>26.668039</td>
          <td>0.170565</td>
          <td>26.002266</td>
          <td>0.156195</td>
          <td>25.862519</td>
          <td>0.254954</td>
          <td>25.223387</td>
          <td>0.321652</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.842736</td>
          <td>1.049038</td>
          <td>26.353496</td>
          <td>0.141706</td>
          <td>26.171288</td>
          <td>0.108813</td>
          <td>25.522968</td>
          <td>0.100885</td>
          <td>25.218844</td>
          <td>0.145330</td>
          <td>24.733212</td>
          <td>0.211143</td>
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
          <td>0.890625</td>
          <td>26.712876</td>
          <td>0.443846</td>
          <td>26.384812</td>
          <td>0.125384</td>
          <td>25.970018</td>
          <td>0.076796</td>
          <td>25.271788</td>
          <td>0.067570</td>
          <td>25.110757</td>
          <td>0.111769</td>
          <td>25.447077</td>
          <td>0.320946</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.645392</td>
          <td>1.503199</td>
          <td>27.484378</td>
          <td>0.314946</td>
          <td>27.268136</td>
          <td>0.234966</td>
          <td>26.885920</td>
          <td>0.270476</td>
          <td>26.313560</td>
          <td>0.307922</td>
          <td>25.636484</td>
          <td>0.372895</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.007061</td>
          <td>0.268534</td>
          <td>25.898340</td>
          <td>0.088074</td>
          <td>24.804497</td>
          <td>0.029726</td>
          <td>23.841672</td>
          <td>0.020929</td>
          <td>23.124458</td>
          <td>0.021145</td>
          <td>22.791115</td>
          <td>0.035323</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.144361</td>
          <td>1.277746</td>
          <td>29.292078</td>
          <td>1.259672</td>
          <td>27.095691</td>
          <td>0.251521</td>
          <td>26.446000</td>
          <td>0.234329</td>
          <td>26.391499</td>
          <td>0.399956</td>
          <td>25.973918</td>
          <td>0.583882</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.913076</td>
          <td>0.515507</td>
          <td>25.720039</td>
          <td>0.070099</td>
          <td>25.414223</td>
          <td>0.046977</td>
          <td>24.819459</td>
          <td>0.045292</td>
          <td>24.400314</td>
          <td>0.059845</td>
          <td>23.821003</td>
          <td>0.080856</td>
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
          <td>2.147172</td>
          <td>26.856094</td>
          <td>0.517567</td>
          <td>26.513547</td>
          <td>0.149903</td>
          <td>26.087062</td>
          <td>0.092130</td>
          <td>25.932604</td>
          <td>0.130973</td>
          <td>25.528848</td>
          <td>0.173074</td>
          <td>25.642732</td>
          <td>0.402258</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.568049</td>
          <td>0.817172</td>
          <td>27.602338</td>
          <td>0.350050</td>
          <td>26.891952</td>
          <td>0.173938</td>
          <td>26.698274</td>
          <td>0.235417</td>
          <td>26.277476</td>
          <td>0.303401</td>
          <td>25.009776</td>
          <td>0.228397</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.247335</td>
          <td>0.671442</td>
          <td>27.223769</td>
          <td>0.265192</td>
          <td>27.009491</td>
          <td>0.198192</td>
          <td>26.664987</td>
          <td>0.236458</td>
          <td>25.834933</td>
          <td>0.217648</td>
          <td>25.850188</td>
          <td>0.458581</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.721978</td>
          <td>0.947287</td>
          <td>27.700708</td>
          <td>0.407895</td>
          <td>26.902153</td>
          <td>0.192841</td>
          <td>25.739164</td>
          <td>0.114899</td>
          <td>25.870492</td>
          <td>0.238504</td>
          <td>25.898027</td>
          <td>0.503303</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.783693</td>
          <td>0.946867</td>
          <td>26.436355</td>
          <td>0.135536</td>
          <td>26.124582</td>
          <td>0.091498</td>
          <td>25.575703</td>
          <td>0.092048</td>
          <td>25.175345</td>
          <td>0.122856</td>
          <td>24.720747</td>
          <td>0.183288</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
