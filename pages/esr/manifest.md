# ESR App
** Update the ESR App for better response time and reliability using the most recent updates to the frames component. We must also add functionality to combine multiple series for most pages. Data for most pages should now originate from the dcc.Store as opposed to fetching new keys**
### Home Updates
- Update home so to contain a dcc.Store item with the id `esr-df-store` to store the downloaded export keys
- Create a dropdown called `esr-commodity-dd` to select which commodity to load into `esr-df-store`
- Give `esr-df-store` a callback which updates based on the commodity selected, load it into the dataframe, and then return the data as a dict with `orient='records'`
- Create another `dcc.Store(id='esr-options-store')` which contains the values for the layout menu options
- Dynamically generate country menu options and pass them to the new page based on the selected commodity
### Sales Trend updates. 
- Update the menu to contain a column dropdown for each of the charts provided in sales trends
- Update the menu to deal with country settings in a way that allows you to select and display individual countries or sum them together
- Change charts to be updated from the `esr-df-store` data source ,the column dropdown, and the country settings

### Commitment Analysis Updates

- Change callback functions (callbacks/esr) to update frame 0 chart_0 from the store and column select. 
- Change callback functions to update the remainder of the charts from the commitment_vs_shipment analysis, with the charts displaying the columns `sales_backlog`, `fulfillment_rate`, and `commitment_utilization`
- Add the same type of country functionality as stated in the sales trend page which allows multiple or single country selection, displaying them together or separately and labeled

### Country Analysis updates
- Create the ability to dynamically generate country menu based on commodity selection, as well as sum multiple countries together
- Overlay past MARKET YEARS of the selected metric from the commitment metric selected, selected with start year and end year 
- Be sure to sum the past market years as well in order to 

### Seasonal Analysis
- Add a new dropdown to select a market year, 
- Create an overlay for a selected market year for the seasonal analysis chart. 
- Create a second new dropdown which is a selection to difference the selected year from the new chart
- Create a new callback to perform this update

### Comparative analysis 
- Create a 


