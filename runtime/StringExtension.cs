public static class StringExtension
{
    public static string GenerateFloatTable(float[,] floatArray, bool includeHeaders = false, string linePrepend="", string lineAppend="")
    {
        int rows = floatArray.GetLength(0);
        int columns = floatArray.GetLength(1);

        // String builder to construct the table
        System.Text.StringBuilder sb = new System.Text.StringBuilder();

        // Optionally add headers
        if (includeHeaders)
        {
            sb.Append(linePrepend);
            for (int j = 0; j < columns; j++)
            {
                sb.Append("Column " + (j + 1) + "\t");
            }
            sb.Append(lineAppend);
            sb.AppendLine();
        }

        // Add data rows
        for (int i = 0; i < rows; i++)
        {
            sb.Append(linePrepend);
            for (int j = 0; j < columns; j++)
            {
                sb.Append(floatArray[i, j].ToString() + "\t"); 
            }
            sb.Append(lineAppend);
            sb.AppendLine();
        }

        return sb.ToString();
    }
}